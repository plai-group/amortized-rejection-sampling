import numpy as np
import torch
from tqdm import tqdm
import matplotlib as mpl


# https://gist.github.com/thriveth/8560036
color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
               '#f781bf', '#a65628', '#984ea3',
               '#999999', '#e41a1c', '#dede00']

labels_dict = {"ic": "IC",
               "prior": "Prior",
               "ars-1": r"$\mathrm{ARS}_{M=1}$",
               "ars-2": r"$\mathrm{ARS}_{M=2}$",
               "ars-5": r"$\mathrm{ARS}_{M=5}$",
               "ars-10": r"$\mathrm{ARS}_{M=10}$",
               "ars-20": r"$\mathrm{ARS}_{M=20}$",
               "ars-50": r"$\mathrm{ARS}_{M=50}$",
               "biased": "Biased",
               "gt": "Groundtruth",
               "is": "IS",
               "collapsed": "Collapsed"}

color_dict = {'gt': color_cycle[0],
              'prior': color_cycle[5],
              'ic': color_cycle[2],
              'biased': color_cycle[3],
              'ars-1': color_cycle[4],
              'ars-2': color_cycle[1],
              'ars-5': color_cycle[7],
              'ars-10': color_cycle[6],
              'ars-100': color_cycle[8],
              'ars-50': color_cycle[8],
              'is': color_cycle[8],
              'ars-20': "C1",
              "collapsed": color_cycle[7]}


########################################
## matplotlib style and configs       ##
########################################
def setup_matplotlib():
    import seaborn as sns
    # mpl.use('Agg')
    # plt.style.use('classic')
    # sns.set(font_scale=1.5)
    sns.set_style('white')
    sns.color_palette('colorblind')
    nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
    mpl.rcParams.update(nice_fonts)


def set_size(width, fraction=1, subplots=(1, 1)):
    # https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    elif width == 'pnas':
        width_pt = 246.09686
    elif width == 'aistats22':
        width_pt = 487.8225
    else:
        width_pt = width

    # Width of figure
    fig_width_pt = width_pt * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


class OOMFormatter(mpl.ticker.ScalarFormatter):
    """OrderOfMagnitude formatter
    
    Source:
    https://stackoverflow.com/questions/42656139/set-scientific-notation-with-fixed-exponent-and-significant-digits-for-multiple
    """
    def __init__(self, order=0, fformat="%1.1f", *args, **kwargs):
        self.oom = order
        self.fformat = fformat
        mpl.ticker.ScalarFormatter.__init__(self,*args, **kwargs)
    def _set_order_of_magnitude(self):
        super()._set_order_of_magnitude()
        self.orderOfMagnitude = self.oom


def add_center_aligned_legend(fig, handles, ncol, **kwargs):
    nlines = len(handles)
    leg1 = fig.legend(handles=handles[:nlines//ncol*ncol], ncol=ncol, **kwargs)
    if nlines % ncol != 0:
        fig.add_artist(leg1)
        leg2 = fig.legend(handles=handles[nlines//ncol*ncol:], ncol=nlines-nlines//ncol*ncol)
        leg2.remove()
        leg1._legend_box._children.append(leg2._legend_handle_box)
        leg1._legend_box.stale = True


########################################
## Loading from disk                  ##
########################################
def load_log_weights(log_weights_root, iw_mode):
    """Loads the log_weights from the disk. It assumes a file structure of <log_weights_root>/<iw_mode>/*.npy
    of mulyiple npy files. This function loads all the weights in a single numpy array, concatenating all npy files.
    Finally, it caches the result in a file stored at <log_weights_root>/<iw_mode>.npy
    In the further calls, it reuses the cached file.

    Args:
        log_weights_root (str or pathlib.Path)
        iw_mode (str)

    Returns:
        np.ndarray: log importance weights
    """
    agg_weights_file = log_weights_root / f"{iw_mode}.npy"
    agg_weights_dir = log_weights_root / iw_mode
    assert agg_weights_dir.exists() or agg_weights_file.exists()
    if not agg_weights_file.exists():
        log_weights = np.concatenate(
            [np.load(weight_file) for weight_file in agg_weights_dir.glob("*.npy")])
        np.save(agg_weights_file, log_weights)
    else:
        log_weights = np.load(agg_weights_file)
    print(f"{log_weights_root} / {iw_mode} has {len(log_weights):,} traces")
    return log_weights


########################################
## Estimators and metrics             ##
########################################
def _compute_estimator_helper(log_weights, dx, estimator_func, **kwargs):
    """A helper function for computing the plotting data. It generates the
    x-values and y-values of the plot. x-values is an increasing sequence of
    integers, with incremens of dx and ending with N. y-values is a TxK tensor
    where T is the number of trials and K is the size of x-values. The j-th
    column of y-values is the estimator applied to the log_weights up to the
    corresponding x-value.

    Args:
        log_weights (torch.FloatTensor of shape TxN): All the log importance weights
            of a particular experiment.
        dx (int): different between points of evaluating the estimator.
        estimator_func (function): the estimator function that operates on a tensor
            of shape Txn where n <= N.
        **kwargs: optional additional arguments to the estimator function
    """
    (T, N) = log_weights.shape
    xvals = _get_xvals(end=N, dx=dx)
    yvals_all = [estimator_func(log_weights[:, :x], **kwargs) for x in xvals]
    yvals_all = torch.stack(yvals_all, dim=1)
    return xvals, yvals_all


def _get_xvals(end, dx):
    """Returns a integer numpy array of x-values incrementing by "dx"
    and ending with "end".

    Args:
        end (int)
        dx (int)
    """
    arange = np.arange(0, end-1+dx, dx, dtype=int)
    xvals = arange[1:]
    return xvals


def _log_evidence_func(arr):
    """Returns an estimate of the log evidence from a set of log importance wegiths
    in arr. arr has shape TxN where T is the number of trials and N is the number
    of samples for estimation.

    Args:
        arr (torch.FloatTensor of shape TxN): log importance weights

    Returns:
        A tensor of shape (T,) representing the estimates for each set of sampels.
    """
    T, N = arr.shape
    log_evidence = torch.logsumexp(arr, dim=1) - np.log(N)
    return log_evidence


def _ess_func(arr):
    """Effective sample size (ESS)"""
    a = torch.logsumexp(arr, dim=1) * 2
    b = torch.logsumexp(2 * arr, dim=1)
    return torch.exp(a - b)


def _ess_inf_func(arr):
    """ESS-infinity (Q_n)"""
    a = torch.max(arr, dim=1)[0]
    b = torch.logsumexp(arr, dim=1)
    return torch.exp(a - b)


def get_evidence_estimate(log_weights, dx):
    return _compute_estimator_helper(log_weights, estimator_func=lambda x: _log_evidence_func(x).exp(), dx=dx)


def get_log_evidence_estimate(log_weights, dx):
    return _compute_estimator_helper(log_weights, estimator_func=_log_evidence_func, dx=dx)


def get_ess(log_weights, dx):
    return _compute_estimator_helper(log_weights, estimator_func=_ess_func, dx=dx)


def get_ness(log_weights, dx):
    """Normalized ESS (ESS / N)"""
    xvals, yvals = get_ess(log_weights, dx=dx)
    return xvals, yvals / xvals


def get_qn(log_weights, dx):
    return _compute_estimator_helper(log_weights, estimator_func=_ess_inf_func, dx=dx)


########################################
## Plotting functions                 ##
########################################
def _lineplot_helper(*, name, func, ax, log_weights_dict, iw_mode_list, dx, bias=None, **kwargs):
    """A helper function for making the line functions of the paper.

    Args:
        name (string): Metric name. Used for logging only.
        func (function): The metric computation function. Should be a function that takes in log_weights and dx
            and returns x-values and y-values. Any additional arguments in kwargs will be passed to this function.
        ax (matplotlib.axes): A matrplotlib ax object in which the plot should be drawn.
        log_weights_dict (dict): A dictionary of the form {iw_mode: log_imprtance_weights as a TxN tensor}
        iw_mode_list (list): An ordered list of iw modes specifying the order of drawing the lines.
        dx (int): The distance between consequent x-values.
        bias (float, optional): If not None, shifts all the line's y-values according to it. Defaults to None.
    """
    for iw_mode in tqdm(iw_mode_list, desc=name):
        if iw_mode not in log_weights_dict:
            print(f"Skipping {iw_mode}.")
            continue
        log_weights = torch.tensor(log_weights_dict[iw_mode])

        label = labels_dict[iw_mode]
        color = color_dict[iw_mode]

        xs, ys_all = func(log_weights, dx=dx)
        means = ys_all.mean(dim=0)
        stds = ys_all.std(dim=0)
        if bias is not None:
            means -= bias
        ax.plot(xs, means, color=color, label=label)
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.2)
        print(f"> ({name}) {iw_mode, means[-1].item(), stds[-1].item()}")


def plot_evidence(**kwargs):
    _lineplot_helper(name="Evidence plot", func=get_evidence_estimate, **kwargs)


def plot_log_evidence(**kwargs):
    _lineplot_helper(name="Evidence plot", func=get_log_evidence_estimate, **kwargs)


def plot_ness(**kwargs):
    _lineplot_helper(name="NESS plot", func=get_ness, **kwargs)


def plot_qn(**kwargs):
    _lineplot_helper(name="Qn plot", func=get_qn, **kwargs)


def plot_convergence(ax, log_weights_dict, dx, iw_mode_list,
                     qn_threshold, n_splits=10):
    plot_labels = []
    plot_x = []
    for iw_mode in tqdm(iw_mode_list, desc="Convergence plot"):
        if iw_mode not in log_weights_dict:
            print(f"Skipping {iw_mode}.")
            continue
        log_weights = torch.tensor(log_weights_dict[iw_mode])

        label = labels_dict[iw_mode]
        
        xs, qns_all = get_qn(log_weights, dx=dx)

        assert qns_all.shape[0] % n_splits == 0, f"The number of trials ({qns_all.shape[0]}) should be divisible by {n_splits}"

        qns_all = qns_all.reshape(n_splits, qns_all.shape[0] // n_splits, -1)
        qn_means = qns_all.mean(dim=0)
        print(f"> (Convergence plot) {iw_mode, qn_means.mean(dim=0)[-1].item()} out of {log_weights.shape[-1]} samples")
        converged = (qn_means < qn_threshold).cpu().numpy()

        plot_labels.append(label)
        if not converged.any(axis=-1).all(): # Some of them are not converged ever
            plot_x.append([])
        else:
            plot_x.append(converged.argmax(axis=-1) * dx)
    ax.boxplot(plot_x, labels=plot_labels, showmeans=True, meanline=True)


def plot_convergence_2(ax, log_weights_dict, dx, iw_mode_list, qn_threshold):
    # Source: https://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation/33330997
    plot_labels = []
    plot_x = []
    for iw_mode in tqdm(iw_mode_list, desc="Convergence plot"):
        if iw_mode not in log_weights_dict:
            print(f"Skipping {iw_mode}.")
            continue
        log_weights = torch.tensor(log_weights_dict[iw_mode])

        label = labels_dict[iw_mode]
        
        xs, qns_all = get_qn(log_weights, dx=dx)

        assert qns_all.shape[0] % 10 == 0
        qns_all = qns_all.reshape(10, qns_all.shape[0] // 10, -1)
        qn_means = qns_all.mean(dim=0)

        converged = (qn_means < qn_threshold).cpu().numpy()

        plot_labels.append(label)
        if not converged.any(axis=-1).all(): # Some of them are not converged ever
            plot_x.append([])
        else:
            plot_x.append(converged.argmax(axis=-1) * dx)
    xvals = [i for i in range(len(plot_x)) if plot_x[i] != []]
    x = np.stack([x for x in plot_x if x != []])
    mins = x.min(axis=1)
    maxes = x.max(axis=1)
    means = x.mean(axis=1)
    std = x.std(axis=1)

    # create stacked errorbars:
    ax.errorbar(xvals, means, std, fmt='ok', lw=3)
    ax.errorbar(xvals, means, [means - mins, maxes - means],
                fmt='.k', ecolor='gray', lw=1)

    ax.set_xticks(np.arange(len(plot_x)))
    ax.set_xticklabels(plot_labels)