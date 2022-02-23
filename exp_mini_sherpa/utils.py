import torch
import numpy as np
import os
import pyprob
from pyprob.util import to_tensor
from pyprob.distributions import Distribution, Uniform

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus']=False
from matplotlib import cm
#colors = [cm.inferno(x) for x in np.linspace(0, 1, 5)]
#colors_inferno = [cm.inferno(x) for x in np.linspace(0, 1, 5)]

class CustomDistribution(Distribution):

    def __init__(self, scale):
        self.scale = to_tensor(scale)
        
        super().__init__(name='Categorical', address_suffix='Categorical')

    def __repr__(self):
        return 'Categorical({})'.format(self.scale)

    def log_prob(self, value, sum=False):
        lp = torch.log(torch.abs(value)) - 2*torch.log(self.scale)
        return torch.sum(lp) if sum else lp

    def pdf(self, x):
         return 1/self.scale*np.abs(x/self.scale)

    def sample(self):
        ymax = self.pdf(self.scale)

        while True:
            xtest   = Uniform(-self.scale, self.scale).sample()
            ysample = Uniform(0, ymax).sample()
            if ysample < self.pdf(xtest):
                return xtest

    @property
    def mean(self):
        if self._mean is None:
            means = torch.stack([d.mean for d in self._distributions])
            if self._batch_length == 0:
                self._mean = torch.dot(self._probs, means)
            else:
                self._mean = torch.diag(torch.mm(self._probs, means))
        return self._mean

    @property
    def variance(self):
        if self._variance is None:
            variances = torch.stack([(d.mean - self.mean).pow(2) + d.variance for d in self._distributions])
            if self._batch_length == 0:
                self._variance = torch.dot(self._probs, variances)
            else:
                self._variance = torch.diag(torch.mm(self._probs, variances))
        return self._variance

min_energy_deposit = 0.05  # GeV

channel_names = ['tau-_nu_taunu_ebe-',
                 'tau-_nu_taunu_mubmu-',
                 'tau-_pi-nu_tau']


def create_path(path, directory=False):
    if directory:
        dir = path
    else:
        dir = os.path.dirname(path)
    if not os.path.exists(dir):
        print('{} does not exist, creating'.format(dir))
        try:
            os.makedirs(dir)
        except Exception as e:
            print(e)
            print('Could not create path, potentiall created by another rank in multinode: {}'.format(path))



def pad_final_state_momenta(m, target_rows=3):
    rows = m.shape[0]
    pad = to_tensor(torch.zeros(target_rows - rows, 3).fill_(-9999))
    return torch.cat([m, pad])


def plot_distribution(dist, obs_mode, model, plot_samples=25000, title='', ground_truth_trace=None, file_name=None):
    if dist.length > 0:
        dist_px = dist.map(lambda x: float(x.result[0][0]))
        dist_py = dist.map(lambda x: float(x.result[0][1]))
        dist_pz = dist.map(lambda x: float(x.result[0][2]))
        dist_channel = dist.map(lambda x: int(x.result[1]))
        dist_px_samples = [dist_px.sample() for i in range(plot_samples)]
        dist_py_samples = [dist_py.sample() for i in range(plot_samples)]
        dist_pz_samples = [dist_pz.sample() for i in range(plot_samples)]
        dist_channel_samples = [dist_channel.sample() for i in range(plot_samples)]

        dist_channel_combined = pyprob.distributions.Empirical(dist_channel._values, log_weights=dist_channel._log_weights)
        dist_channel_combined_values = dist_channel_combined._values
        dist_channel_combined_weights = [float(torch.exp(w)) for w in dist_channel_combined._log_weights]
        dist_channel_str = 'Channel probabilities: ' + ', '.join(['{}: {:.3f}'.format(v, w) for v, w in zip(dist_channel_combined_values, dist_channel_combined_weights)])

        fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(25, 10))
        ax1.text(-10, 14, title)
        ax1.text(-10, 12, dist_channel_str)
        ax2.title.set_text('Channel')
        # ax2.text(0.5, 0.5, 'mean={:.3f} (solid), stddev={:.3f}'.format(float(dist_channel.mean), float(dist_channel.stddev)), ha='center', va='center', transform=ax2.transAxes)
        ax3.title.set_text('p_x')
        ax3.text(0.5, 0.5, 'mean={:.3f} (solid), stddev={:.3f}'.format(float(dist_px.mean), float(dist_px.stddev)), ha='center', va='center', transform=ax3.transAxes)
        ax4.title.set_text('p_y')
        ax4.text(0.5, 0.5, 'mean={:.3f} (solid), stddev={:.3f}'.format(float(dist_py.mean), float(dist_py.stddev)), ha='center', va='center', transform=ax4.transAxes)
        ax5.title.set_text('p_z')
        ax5.text(0.5, 0.5, 'mean={:.3f} (solid), stddev={:.3f}'.format(float(dist_pz.mean), float(dist_pz.stddev)), ha='center', va='center', transform=ax5.transAxes)

        _ = ax2.hist(dist_channel_samples, bins=[0.5,1.5,2.5,3.5], density=True, alpha=0.6, color=colors[0])
        ax2.set_xlim([0.5,3.5])
        ax2.set_ylim([0,1])
        ax2.set_xticks([1,2,3])
        # ax2.axvline(float(dist_channel.mean), color='gray', linestyle='solid', linewidth=1)
        _ = ax3.hist(dist_px_samples, bins=np.arange(-0.5,0.5+0.02,0.02), density=True, alpha=0.6, color=colors[1])
        ax3.set_xlim([-0.5,0.5])
        ax3.set_ylim([0,50])
        ax3.axvline(float(dist_px.mean), color='gray', linestyle='solid', linewidth=1)
        _ = ax4.hist(dist_py_samples, bins=np.arange(-0.5,0.5+0.02,0.02), density=True, alpha=0.6, color=colors[2])
        ax4.set_xlim([-0.5,0.5])
        ax4.set_ylim([0,50])
        ax4.axvline(float(dist_py.mean), color='gray', linestyle='solid', linewidth=1)
        _ = ax5.hist(dist_pz_samples, bins=np.arange(10,20+0.2,0.2), density=True, alpha=0.6, color=colors[3])
        ax5.set_xlim([10,20])
        ax5.set_ylim([0,5])
        ax5.axvline(float(dist_pz.mean), color='gray', linestyle='solid', linewidth=1)

        if ground_truth_trace is not None:
            ax2.axvline(float(ground_truth_trace.result[1]), color='gray', linestyle='dashed', linewidth=2)
            ax3.axvline(float(ground_truth_trace.result[0][0]), color='gray', linestyle='dashed', linewidth=2)
            ax4.axvline(float(ground_truth_trace.result[0][1]), color='gray', linestyle='dashed', linewidth=2)
            ax5.axvline(float(ground_truth_trace.result[0][2]), color='gray', linestyle='dashed', linewidth=2)
            # ax2.text(0.5, 0.4, 'ground_truth={:.3f} (dashed)'.format(float(ground_truth_trace.result[1])), ha='center', va='center', transform=ax2.transAxes)
            ax3.text(0.5, 0.4, 'ground_truth={:.3f} (dashed)'.format(float(ground_truth_trace.result[0][0])), ha='center', va='center', transform=ax3.transAxes)
            ax4.text(0.5, 0.4, 'ground_truth={:.3f} (dashed)'.format(float(ground_truth_trace.result[0][1])), ha='center', va='center', transform=ax4.transAxes)
            ax5.text(0.5, 0.4, 'ground_truth={:.3f} (dashed)'.format(float(ground_truth_trace.result[0][2])), ha='center', va='center', transform=ax5.transAxes)

        dist_final_state_momenta = dist.map(lambda x: pad_final_state_momenta(x.result[2]))
        dist_final_state_momenta_m1x = dist_final_state_momenta.map(lambda x: x[0,0])
        dist_final_state_momenta_m1y = dist_final_state_momenta.map(lambda x: x[0,1])
        dist_final_state_momenta_m1z = dist_final_state_momenta.map(lambda x: x[0,2])
        dist_final_state_momenta_m1x_samples = [float(dist_final_state_momenta_m1x.sample()) for i in range(plot_samples)]
        dist_final_state_momenta_m1y_samples = [float(dist_final_state_momenta_m1y.sample()) for i in range(plot_samples)]
        dist_final_state_momenta_m1z_samples = [float(dist_final_state_momenta_m1z.sample()) for i in range(plot_samples)]

        dist_final_state_momenta_m2x = dist_final_state_momenta.map(lambda x: x[1,0]).filter(lambda x: float(x) != -9999)
        dist_final_state_momenta_m2y = dist_final_state_momenta.map(lambda x: x[1,1]).filter(lambda x: float(x) != -9999)
        dist_final_state_momenta_m2z = dist_final_state_momenta.map(lambda x: x[1,2]).filter(lambda x: float(x) != -9999)
        if dist_final_state_momenta_m2x.length > 0:
            dist_final_state_momenta_m2x_samples = [float(dist_final_state_momenta_m2x.sample()) for i in range(plot_samples)]
            dist_final_state_momenta_m2y_samples = [float(dist_final_state_momenta_m2y.sample()) for i in range(plot_samples)]
            dist_final_state_momenta_m2z_samples = [float(dist_final_state_momenta_m2z.sample()) for i in range(plot_samples)]
        else:
            dist_final_state_momenta_m2x_samples = []
            dist_final_state_momenta_m2y_samples = []
            dist_final_state_momenta_m2z_samples = []

        dist_final_state_momenta_m3x = dist_final_state_momenta.map(lambda x: x[2,0]).filter(lambda x: float(x) != -9999)
        dist_final_state_momenta_m3y = dist_final_state_momenta.map(lambda x: x[2,1]).filter(lambda x: float(x) != -9999)
        dist_final_state_momenta_m3z = dist_final_state_momenta.map(lambda x: x[2,2]).filter(lambda x: float(x) != -9999)
        if dist_final_state_momenta_m3x.length > 0:
            dist_final_state_momenta_m3x_samples = [float(dist_final_state_momenta_m3x.sample()) for i in range(plot_samples)]
            dist_final_state_momenta_m3y_samples = [float(dist_final_state_momenta_m3y.sample()) for i in range(plot_samples)]
            dist_final_state_momenta_m3z_samples = [float(dist_final_state_momenta_m3z.sample()) for i in range(plot_samples)]
        else:
            dist_final_state_momenta_m3x_samples = []
            dist_final_state_momenta_m3y_samples = []
            dist_final_state_momenta_m3z_samples = []

        _ = ax8.hist([dist_final_state_momenta_m1x_samples, dist_final_state_momenta_m2x_samples, dist_final_state_momenta_m3x_samples], bins=20, histtype='bar', density=True, alpha=0.6)
        ax8.title.set_text('Final state particles p_x')
        _ = ax9.hist([dist_final_state_momenta_m1y_samples, dist_final_state_momenta_m2y_samples, dist_final_state_momenta_m3y_samples], bins=20, histtype='bar', density=True, alpha=0.6)
        ax9.title.set_text('Final state particles p_y')
        _ = ax10.hist([dist_final_state_momenta_m1z_samples, dist_final_state_momenta_m2z_samples, dist_final_state_momenta_m3z_samples], bins=20, histtype='bar', density=True, alpha=0.6)
        ax10.title.set_text('Final state particles p_z')

        if ground_truth_trace is not None:
            ground_truth_final_state_momenta = pad_final_state_momenta(ground_truth_trace.result[2])
            for i in range(ground_truth_final_state_momenta.size(0)):
                if float(ground_truth_final_state_momenta[i, 0]) != -9999:
                    ax8.axvline(float(ground_truth_final_state_momenta[i, 0]), color='gray', linestyle='dashed', linewidth=2)
                if float(ground_truth_final_state_momenta[i, 1]) != -9999:
                    ax9.axvline(float(ground_truth_final_state_momenta[i, 1]), color='gray', linestyle='dashed', linewidth=2)
                if float(ground_truth_final_state_momenta[i, 2]) != -9999:
                    ax10.axvline(float(ground_truth_final_state_momenta[i, 2]), color='gray', linestyle='dashed', linewidth=2)

        dist_mean_n_deposits = dist.map(lambda x: x.result[3])
        mean_n_deposits = dist_mean_n_deposits.mean.cpu().view(20,20).data.numpy()
        xc,yc = np.meshgrid(*model._binning)
        c = ax1.pcolor(xc,yc,mean_n_deposits.T,cmap='viridis')
        ticks = [-10,-5,0,5,10]
        ax1.set_xticks(ticks)
        ax1.set_yticks(ticks)
        ax1.title.set_text('Simulated calorimeter mean')
        fig.colorbar(c, ax=ax1)

        if ground_truth_trace is not None:
            observation = ground_truth_trace.samples_observed[0].distribution.mean.view(20, 20)
            xc,yc = np.meshgrid(*model._binning)
            c = ax6.pcolor(xc,yc,observation.data.cpu().numpy().T,cmap='viridis')
            ticks = [-10,-5,0,5,10]
            ax6.set_xticks(ticks)
            ax6.set_yticks(ticks)
            ax6.title.set_text('Observed calorimeter')
            fig.colorbar(c, ax=ax6)

            mean_n_deposits = to_tensor(mean_n_deposits)
            obs_log_prob = (mean_n_deposits.log() * observation) - mean_n_deposits - (observation + 1).lgamma()
            xc,yc = np.meshgrid(*model._binning)
            c = ax7.pcolor(xc,yc,obs_log_prob.data.numpy().T,cmap='viridis')
            ticks = [-10,-5,0,5,10]
            ax7.set_xticks(ticks)
            ax7.set_yticks(ticks)
            ax7.title.set_text('Log-likelihood of observed calorimeter')
            fig.colorbar(c, ax=ax7)

        if file_name is not None:
            plt.savefig(file_name, bbox_inches='tight')


def get_trace_text(trace):
    observed_variable = trace.named_variables['calorimeter_n_deposits']
    data = observed_variable.value.cpu().view(20, 20).data.numpy()
    named_variables_dict = {v.name: v for v in trace.variables if v.name is not None}
    final_state_momenta = trace.result[2].cpu().data.numpy()
    trace_str = str(trace)

    data_flat = data.reshape(-1)
    non_zero = np.count_nonzero(data_flat)
    data_flat_min = min(data_flat) * min_energy_deposit
    data_flat_max = max(data_flat) * min_energy_deposit
    data_flat_total = np.sum(data_flat) * min_energy_deposit

    particles = final_state_momenta.reshape(-1, 3)

    trace_text = []
    trace_text.append(trace_str)
    trace_text.append('p_x      : {}'.format(float(named_variables_dict['mother_momentum_x'].value)))
    trace_text.append('p_y      : {}'.format(float(named_variables_dict['mother_momentum_y'].value)))
    trace_text.append('p_z      : {}'.format(float(named_variables_dict['mother_momentum_z'].value)))
    channel = int(named_variables_dict['channel_index'].value)
    trace_text.append('channel  : {} ({})'.format(channel, channel_names[channel]))
    trace_text.append('particles: {}'.format(particles.shape[0]))
    trace_text.append('calorimeter:')
    trace_text.append('non-zero pixels: {:,}/{:,} ({:.2f}%)'.format(non_zero, len(data_flat), 100*non_zero/len(data_flat)))
    trace_text.append(' min   : {:.3f} GeV'.format(data_flat_min))
    trace_text.append(' max   : {:.3f} GeV'.format(data_flat_max))
    trace_text.append(' total : {:.3f} GeV'.format(data_flat_total))
    trace_text.append(f'Trace length = {len(trace.variables)}')
    trace_text.append(f'Number of rejected samples = {sum([int(e.iteration) for e in trace.rs_entries])}')
    trace_text = '\n'.join(trace_text)

    return trace_text


def plot_trace(trace, binning, surface_z, fig, axes, mode="2d"):
    assert mode in ["2d", "3d"], f"mode is expected to be either 2d or 3d (got {mode})."

    observed_variable = trace.named_variables['calorimeter_n_deposits']
    data_mean = observed_variable.distribution.mean.cpu().view(20, 20).data.numpy()
    data = observed_variable.value.cpu().view(20, 20).data.numpy()
    final_state_momenta = trace.result[2].cpu().data.numpy()

    xc, yc = np.meshgrid(*binning)
    if mode == "2d":
        for ax, _data in zip((axes[0], axes[1]), (data_mean, data)):
            ax.set_aspect('equal', adjustable='box')
            ax.pcolor(xc, yc, _data.T, cmap='viridis')
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    elif mode == "3d":
        ax = axes
        colors = ['r', 'b', 'g', 'y', 'k']
        ax.set_axis_off()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        for v, c in zip(final_state_momenta, colors):
            traj = np.array([(np.asarray(v)*t).tolist() for t in np.linspace(0, abs(surface_z*1.5/v[2]))])
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c=c, zorder=10)
        ax.set_zlim(0, surface_z)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.plot_surface(xc, yc, np.ones_like(xc)*surface_z, cstride=1, rstride=1, facecolors=plt.cm.viridis(data_mean.T/np.max(data_mean.T)), shade=False, zorder=0, alpha=0.8)
        ax.view_init(45, -45)
        #ax.tick_params(axis='both', which='both', pad=-2)