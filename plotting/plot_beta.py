import torch
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
import json
import os
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import utils

#IW_MODE_LIST = ["ic", "ars-1", "ars-2", "ars-10", "prior"]
IW_MODE_LIST = ["ic", "ars-1", "ars-10", "prior"]

N_TRACES = 100 * 10 * 1000
TRIALS = 100
X_MAX = N_TRACES // TRIALS
NUM_TRACES = TRIALS * X_MAX
dx = 100


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", type=str, default="plots")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--qn_threshold", type=float, default=0.01)
    parser.add_argument("--exp_part", default="a", choices=["a", "b"],
                        help="Which experiment part to make (i.e. plot a or b)")
    args = parser.parse_args()

    # Initialize the figure
    args.out = Path(args.out) / f"exp_beta"
    utils.setup_matplotlib()
    figsize = utils.set_size("aistats22", fraction=0.5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Load weights
    exp_name = f"beta_{args.exp_part}"
    log_weights_dict = {}
    for iw_mode in IW_MODE_LIST:
        log_weights_dict[iw_mode] = {}
        for num_obs in range(1, 31):
            log_weights_root = Path("../../out/beta/log_importance_weights") / f"{exp_name}-{num_obs}"
            # Load the weights
            log_weights = utils.load_log_weights(log_weights_root, iw_mode)
            log_weights = log_weights[:NUM_TRACES]
            if len(log_weights) != NUM_TRACES:
                print(f"Traces are not enough ({exp_name}, {iw_mode}: {len(log_weights)})")
            else:
                log_weights_dict[iw_mode][num_obs] = log_weights.reshape(TRIALS, -1)

    # Convergence plot
    iw_mode_list = IW_MODE_LIST
    qn_threshold = args.qn_threshold
    for iw_mode in tqdm(iw_mode_list, desc="Convergence plot"):
        if iw_mode not in log_weights_dict:
            print(f"Skipping {iw_mode}.")
            continue
        xs, means, stds = [], [], []
        label = utils.labels_dict[iw_mode]
        color = utils.color_dict[iw_mode]
        for num_obs in range(1, 31):
            log_weights = torch.tensor(log_weights_dict[iw_mode][num_obs])
            _, qns_all = utils.get_qn(log_weights, dx=dx)
            qn_means = qns_all.mean(axis=0)
            converged = (qn_means < args.qn_threshold).cpu().numpy()
            if np.any(converged): # If not converged, do not include the point in the plot
                xs.append(num_obs)
                means.append(np.argmax(converged) * dx)
        ax.plot(xs, means, color=color, label=label, marker=".",
                alpha=0.8)

    # Ax labels and ticks
    ax.tick_params(axis='both', which='major', length=4, left=True, bottom=True)
    ax.set_xlim(left=0, right=30)
    # Ax labels
    ax.set_ylabel(r"$K_{\mathrm{converge}}$")
    ax.set_xlabel(r"$n$")
    
    ax.grid()

    handles, labels = ax.get_legend_handles_labels()
    utils.add_center_aligned_legend(fig, handles, ncol=5,
                                    loc='upper center',shadow=False,
                                    frameon=True, borderpad=0.2)
    fig.tight_layout(rect=(0,0,1,0.87))
    if args.debug:
        plt.show()
    else:
        path = str(args.out) + f"_{args.exp_part}.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved to {path}")
    quit()