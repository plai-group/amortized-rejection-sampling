import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import utils

#IW_MODE_LIST = ["prior", "ic", "ars-1", "ars-2", "ars-10", "biased", "gt"]
#IW_MODE_LIST = ["ic", "ars-1", "ars-2", "prior", "biased"]
IW_MODE_LIST = ["ars-1", "ars-2", "ars-10", "ic", "biased"]

TRIALS = 100
X_MAX = 100 * 1000
N_TRACES = X_MAX * TRIALS
NUM_TRACES = TRIALS * X_MAX
dx = 1000
dx_convergence = 100


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", type=str, default="plots")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--qn_threshold", type=float, default=0.01)
    parser.add_argument("--exp_part", default="a", choices=["a", "b"],
                        help="Which experiment part to make (i.e. plot a or b)")
    args = parser.parse_args()

    args.out = Path(args.out) / f"exp_marsaglia"
    utils.setup_matplotlib()
    
    # Initialize the figure
    figsize = utils.set_size("aistats22", subplots=(1, 3))
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0], figsize[1] * 1.3))

    # Load weights
    exp_name = f"marsaglia_{args.exp_part}"
    config_file = Path("../configs/") / f"{exp_name}.json"
    log_weights_root = Path("../out/marsaglia/log_importance_weights") / exp_name
    collapsed_log_weights_root = Path("../out/marsaglia_collapsed/log_importance_weights") / exp_name
    # Load the config file
    with open(config_file, "r") as f:
        config = json.load(f)
    log_weights_dict = {}
    for iw_mode in IW_MODE_LIST:
        # Load the weights
        log_weights = utils.load_log_weights(log_weights_root, iw_mode)
        log_weights = log_weights[:NUM_TRACES]
        if len(log_weights) != NUM_TRACES:
            print(f"Traces are not enough ({exp_name}, {iw_mode}: {len(log_weights)})")
        else:
            log_weights_dict[iw_mode] = log_weights.reshape(TRIALS, -1)

    # Evidence plot
    ax = axes[0]
    utils.plot_evidence(ax=ax, log_weights_dict=log_weights_dict,
                        dx=dx, iw_mode_list=IW_MODE_LIST, bias=config["evidence"])
    # NESS plot
    ax = axes[1]
    utils.plot_ness(ax=ax, log_weights_dict=log_weights_dict,
                    dx=dx, iw_mode_list=IW_MODE_LIST)
    # Convergence plot
    ax = axes[2]
    utils.plot_convergence(ax=ax,
                           log_weights_dict={k: v for (k,v) in log_weights_dict.items() if k != "ic"},
                           dx=dx_convergence, iw_mode_list=IW_MODE_LIST,
                           qn_threshold=args.qn_threshold)

    # Ax labels and ticks
    for ax in axes:
        ax.tick_params(axis='both', which='minor', length=2, left=True, bottom=True)
        ax.tick_params(axis='both', which='major', length=4, left=True, bottom=True)
    for ax in axes[:2]:
        ax.xaxis.set_major_formatter(utils.OOMFormatter(order=3))
        ax.set_xlim(left=0, right=100000)
        ax.grid()
    axes[0].set_yscale('symlog', subs=[1,2,3,4,5,6,7,8,9], linthresh=0.01, linscale=0.5)
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axes[2].tick_params(axis="x", labelrotation=45)
    # Ax labels
    axes[0].set_ylabel(r"$\hat{p}(y) - p(y)$")
    axes[1].set_ylabel(r"$\mathrm{ESS} / N$")
    axes[2].set_ylabel(r"$K_{\mathrm{converge}}$")
    if args.exp_part != "a":
        axes[0].set_xlabel("Number of draws (N)", labelpad=14)
        axes[1].set_xlabel("Number of draws (N)", labelpad=14)

    handles, labels = axes[0].get_legend_handles_labels()
    utils.add_center_aligned_legend(fig, handles, ncol=6,
                                    loc='upper center',shadow=False,
                                    frameon=True, borderpad=0.2)
    fig.tight_layout(rect=(0,0,1,0.98))
    if args.debug:
        plt.show()
    else:
        path = str(args.out) + f"_{args.exp_part}.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved to {path}")