"""
The implementation of this model (Mini-SHERPA) is provided by Lukas Heinrich
http://www.lukasheinrich.com/
https://github.com/lukasheinrich
"""
import pyprob
from pyprob import Model, InferenceEngine, InferenceNetwork, PriorInflation
from pyprob.distributions import Distribution, Empirical, Mixture, Uniform, Normal, Categorical, Poisson
from pyprob.util import to_tensor, to_numpy, get_time_stamp

import itertools
import torch
import numpy as np
import scipy
import scipy.stats
import scipy.linalg as la


custom_pdf = lambda x, scale: 1/scale*np.abs(x/scale)


def M(axis, theta):
    return la.expm(np.cross(np.eye(3), axis/la.norm(axis)*theta))


def rotations(phi,theta):
    M1 = M([0,0,1], phi)
    M2 = M([0,1,0], theta)
    return lambda v: np.dot(M2, np.dot(M1,v)), lambda v: np.dot(la.inv(M1),np.dot(la.inv(M2),v))


def to_polar(v):
    x,y,z = v
    r, phi = la.norm(v), np.arctan2(y,x)
    theta = np.arccos(z/r)
    return r, phi, theta


def align(v):
    x,y,z = v
    r, phi, theta = to_polar(v)
    forward, backward = rotations(-phi,-theta)
    aligned =forward(v)
    return forward, backward


def split(v, theta, phi, first_norm):
    fwd, back = align(v)
    a = fwd(v)
    d1 = rotations(-phi,-theta)[1](a*first_norm/np.cos(theta))
    d2 = a - d1
    return  back(d1),  back(d2)


def shower_pars(v, surface_z):
    scaled = (surface_z/v[2])*np.asarray(v)
    loc = scaled[:2]
    sigmas = [1,1]
    return loc, sigmas


def deposit(v, surface_z, single_dep):
    loc, sigmas = shower_pars(v, surface_z)
    if v[2] < 0:
        print('wrong direction')
    if v[2] > 0:
        ndeps = int(la.norm(v)/single_dep)
        return pyprob.sample(Normal([loc for i in range(ndeps)], [sigmas for i in range(ndeps)]), control=False).data.cpu().numpy()
    else:
        return np.empty([0, 2])


def multivar_norm_pdf(x,mean,cov):
    x, mean, cov = np.asarray(x), np.asarray(mean), np.asarray(cov)
    expv = ((x-mean).dot(np.linalg.inv(cov).dot(x-mean)))
    return np.exp(-0.5*expv)/np.sqrt((np.power(2*np.pi,len(x)))*np.linalg.det(cov))


def mean_vals(bincnt, binvol, ener, sfra, locs, covs):
    ws = [E*sf for E,sf in zip(ener,sfra)]
    weights = []
    v = None
    vps = []
    for loc, cov, E, sf in zip(locs, covs, sfra, ener):
        w = E*sf
        #vp = np.asarray([multivar_norm_pdf(bc, loc, cov) for bc in bincnt])
        vp = scipy.stats.multivariate_normal(mean = loc, cov = cov).pdf(bincnt)
        vps.append(vp)
        v = w*vp if v is None else v + w*vp
        weights.append(w)
    v  = v / np.sum(weights)
    v  = v * np.sum(weights) * binvol
    return vps, v


def bin_centers_and_vol(binedges):
    widths  = []
    centers = []
    for axis in binedges:
        axwidths = axis[1:]-axis[:-1]
        assert np.all(axwidths == axwidths[0])
        axcenters = axis[:-1]+axwidths/2
        widths.append(axwidths)
        centers.append(axcenters)
    points = []
    for x,y in itertools.product(*centers):
        points.append([x,y])
    points = np.asarray(points)
    return points, np.prod(widths)


class MiniSherpa(Model):
    def __init__(self,
                 binning=[np.linspace(-10, 10, 21), np.linspace(-10, 10, 21)],
                 surface_z=3, singledep=0.3, calorimeter_noise=0):
        self._binning   = binning
        self._singledep = singledep
        self._calorimeter_noise = calorimeter_noise
        self._bincenters, self._binvol = bin_centers_and_vol(self._binning)

        self._surface_z = surface_z
        super().__init__('Sherpa dummy in Python')

    def stochastic_calo(self, final_state_momenta):
        deps = [deposit(v, self._surface_z, self._singledep) for v in final_state_momenta]
        alldeps = np.concatenate(deps)
        calo_histo, xe, ye = np.histogram2d(alldeps[:, 0], alldeps[:, 1], bins=self._binning)
        calo_histo = calo_histo*self._singledep
        return calo_histo, deps

    def deterministic_calo(self, final_state_momenta, nx, ny):
        sp        = [shower_pars(v, self._surface_z) for v in final_state_momenta]
        locs      = [p[0] for p in sp]
        covs      = [np.square(np.diag(p[1])) for p in sp]
        energies  = [la.norm(v) for v in final_state_momenta]
        sampfracs = [1. for v in final_state_momenta]

        _, smooth = mean_vals(self._bincenters, self._binvol, energies, sampfracs, locs, covs)
        smoothMesh = np.asarray(smooth).T.reshape(ny, nx)
        return smoothMesh

    def split_n(self, v, n):
        splits = []
        rest = v

        thetas = [self.rejection_sample(scale = np.pi/4)[0] for i in range(n-1)]
        phis   = [pyprob.sample(Uniform(0, 2*np.pi)).data.cpu().numpy() for i in range(n-1)]

        for i,(theta,phi) in enumerate(zip(thetas, phis)):
            norm = 1./(n-i)
            a, rest = split(rest, theta, phi, first_norm = norm)
            splits.append(a)
        splits.append(rest)
        return splits

    def rejection_sample(self, scale):
        ymax = custom_pdf(scale,scale)

        ntries = 0
        while True:
            pyprob.rs_start()
            ntries  = ntries+1
            name = 'base'
            xtest   = pyprob.sample(Uniform(-scale, scale), name=name).data.cpu().numpy()
            name = 'acceptance'
            ysample = pyprob.sample(Uniform(0, ymax), name=name).data.cpu().numpy()
            if ysample <= custom_pdf(xtest,scale):
                pyprob.rs_end()
                return xtest, ntries

    def forward(self):
        channel = pyprob.sample(Categorical([1/3, 1/3, 1/3]), name='channel_index').float() + 1
        mother_momentum = [pyprob.sample(Uniform(-0.5, 0.5), name='mother_momentum_x'),
                           pyprob.sample(Uniform(-0.5, 0.5), name='mother_momentum_y'),
                           pyprob.sample(Uniform(10, 20), name='mother_momentum_z')]
        final_state_momenta = self.split_n(list(map(float, mother_momentum)), int(channel))

        smoothMesh = self.deterministic_calo(final_state_momenta, nx=len(self._binning[0])-1, ny=len(self._binning[1])-1)
        mean_n_deposits = to_tensor(smoothMesh / self._singledep).view(-1) + self._calorimeter_noise
        #obs_n_deposits = to_tensor(observation.view(-1))  # observation is assumed to be the number of deposits

        likelihood = Normal(mean_n_deposits, torch.max(torch.sqrt(mean_n_deposits), torch.tensor(0.3)))
        pyprob.observe(likelihood, name='calorimeter_n_deposits')

        return to_tensor(mother_momentum), channel, to_tensor(final_state_momenta), mean_n_deposits

    