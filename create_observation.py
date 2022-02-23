from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import os
from exp_mini_sherpa.utils import get_trace_text, plot_trace
import matplotlib.pyplot as plt
import torch
import pyprob
import numpy as np

from utils import OUTPUTS_ROOT, DotDict, dispatch_model, set_seed
import plotting.utils as plt_utils


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("exp", type=str, choices=["mini_sherpa"])
    parser.add_argument("--arg_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Load the argument file
    assert Path(args.arg_file).exists()
    with open(args.arg_file) as f:
        data = DotDict(json.load(f))
        for k,v in data.items():
            #assert k not in args.__dict__ or getattr(args, k) is None, f"Conflicting argument: {k}"
            setattr(args, k, v)
    print(args)
    if args.seed is not None:
        args.observation.seed = args.seed
    set_seed(args.observation.seed) # Set random seed
    model = dispatch_model(args) # Load the model
    trace = model.sample(trace_mode=pyprob.TraceMode.PRIOR_FOR_INFERENCE_NETWORK) # Draw one trace from prior
    channel = trace.named_variables["channel_index"].value.item()
    print(f"channel = {channel}")
    observed_variable = trace.named_variables["calorimeter_n_deposits"]
    obs = observed_variable.value
    ## Plot the observation ##
    plt_utils.setup_matplotlib()
    (fig_w, fig_h) = plt_utils.set_size("aistats22")
    # 2D plot
    fig_2d, axes = plt.subplots(1, 2, figsize=(fig_w, fig_w / 2), sharey=True)
    plot_trace(trace, model._binning, model._surface_z, fig_2d, axes, mode="2d")
    # 3D plot
    fig_3d, axes = plt.subplots(1, 1, figsize=(fig_w / 2, fig_w / 2))
    plot_trace(trace, model._binning, model._surface_z, fig_3d, axes, mode="3d")
    if args.debug:
        plt.show()
    else:
        trace_text = get_trace_text(trace)
        obs_root = Path("out/") / args.exp / "observations"
        if args.seed is None:
            obs_path = obs_root / args.observation.filename
        else:
            obs_path = obs_root / f"{args.seed}"
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        fig_2d.savefig(f'{obs_path}_2d.pdf', bbox_inches='tight')
        fig_3d.savefig(f'{obs_path}_3d.pdf', bbox_inches='tight')
        with open(f'{obs_path}.txt', 'w') as file:
            file.writelines(trace_text)
        torch.save([trace, model._binning, model._surface_z], f'{obs_path}.trace')
        np.save(f"{obs_path}.npy", obs)
        print(f"Saved the observation at {obs_path}.npy, the trace (and some other info) at {obs_path}.trace its description at {obs_path}.txt, and its visualization at {obs_path}.pdf")
    
    