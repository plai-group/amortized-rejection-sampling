from argparse import ArgumentParser, Namespace
from pathlib import Path
import pickle
import uuid
import json
import os
from pprint import pprint
import pyprob
from pyprob.distributions import Beta
import numpy as np
import scipy.special

from utils import OUTPUTS_ROOT, DotDict, dispatch_model, set_seed


def iw_mode2weighting(iw_mode):
    """Convert iw_mode string to pyprob's importance_weighting

    Args:
        iw_mode (str): either of "prior", "ic", "biased", "gt", or "ars-M"
        where M is the hyper-parameter in ars.
    """
    N = M = None
    if args.iw_mode == "prior":
        importance_weighting = pyprob.ImportanceWeighting.IW0
    elif args.iw_mode == "ic":
        importance_weighting = pyprob.ImportanceWeighting.IW2
    elif args.iw_mode == "biased":
        importance_weighting = pyprob.ImportanceWeighting.IW1
        M = 0
        N = 0
    elif args.iw_mode.startswith("ars-"):
        importance_weighting = pyprob.ImportanceWeighting.IW1
        assert len(args.iw_mode.split("-")) == 2
        num = int(args.iw_mode.split("-")[-1])
        M = num
        N = max(num, 10)
    elif args.iw_mode == "gt":
        importance_weighting = pyprob.ImportanceWeighting.IW1
    elif args.iw_mode.startswith("is"):
        importance_weighting = pyprob.ImportanceWeighting.IW0
    else:
        raise ValueError(f"Unexpected iw_mode: {iw_mode}")
    return importance_weighting, N, M


def get_posterior_kwargs(args):
    kwargs = {}
    observe = None
    if args.exp in ["marsaglia", "marsaglia_collapsed"]:
        observe = {f'obs_{i}': obs for i, obs in enumerate(args.model.observed_list)}
        kwargs["z_p_gt"] = lambda trace: np.pi / 4
        kwargs["z_q_gt"] = lambda trace: args.z_q_gt
    elif args.exp == "beta":
        observe = {f"obs_{i}": True for i in range(args.num_obs)}
        posterior = Beta(args.model.beta + args.num_obs, args.model.beta)
        assert args.proposal in ["posterior_base", "posterior"], f"Unexpected \"proposal\" argument ({args.proposal})"
        kwargs["proposals"] = {"base": posterior}
        if args.iw_mode == "gt":
            z_p = 4**(args.model.beta - 1) * scipy.special.beta(args.model.beta, args.model.beta)
            if args.proposal == "posterior_base":
                z_q = 4**(args.model.beta - 1) / scipy.special.beta(args.model.beta + args.num_obs, args.model.beta) \
                    * scipy.special.beta(2*args.model.beta + args.num_obs - 1, 2*args.model.beta - 1)
            else:
                z_q = 1 / 100
            kwargs["z_p_gt"] = lambda trace: z_p
            kwargs["z_q_gt"] = lambda trace: z_q
    elif args.exp == "mini_sherpa":
        obs_filename = args.observation.filename
        observation_path = Path("out/mini_sherpa/observations") / f"{obs_filename}.npy"
        observe = {'calorimeter_n_deposits': np.load(observation_path)}
        print(f"Observation loaded from {observation_path}")
    else:
        raise ValueError(f"Unexpected argument exp: {args.exp}")
    return observe, kwargs


def main(args):
    # Load the model and initialize observations
    model = dispatch_model(args)
    # A little hack to make using the full posterior as proposal possible for the Beta experiment
    if args.exp == "beta" and args.proposal == "posterior":
        model.perfect_proposal = True
    # Load inference network, if given.
    if args.network_path is not None and args.iw_mode != "is":
        print(f'Loading the inference compulation network from {args.network_path}')
        model.load_inference_network(args.network_path)
        inference_engine = pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK
    else:
        inference_engine = pyprob.InferenceEngine.IMPORTANCE_SAMPLING
    # Prepare pyprob.posterior arguments
    observe, kwargs = get_posterior_kwargs(args)
    importance_weighting, N, M = iw_mode2weighting(args.iw_mode)

    if args.exp in ["marsaglia", "marsaglia_collapsed"]:
        evidence = model.evidence(observe)
        print(f"Evidence = {evidence}")
        if not hasattr(args, "evidence"):
            with open(args.arg_file, "r") as f:
                config = json.load(f)
            config["evidence"] = evidence
            with open(args.arg_file, "w") as f:
                json.dump(config, f, indent=4)
                print(f"Updated {args.arg_file}")
    # Draw traces
    print("-" * 40)
    print("Running posterior inference with the following arguments:")
    print(f"observe = {observe}")
    print(f"Number of traces = {args.num_traces}")
    print(f"inference_engine = {inference_engine}")
    print(f"importance_weighting = {importance_weighting}")
    print(f"num_z_estimate_samples = {N}")
    print(f"num_z_inv_estimate_samples = {M}")
    for k, v in kwargs.items():
        print(f"{k} = {v}")
    traces = model.posterior(observe=observe,
                             num_traces=args.num_traces,
                             inference_engine=inference_engine,
                             importance_weighting=importance_weighting,
                             num_z_estimate_samples=N,
                             num_z_inv_estimate_samples=M,
                             **kwargs)
    log_importance_weights = traces.map(
        lambda trace: trace.log_importance_weight).values
    print("-" * 40)
    return np.array(log_importance_weights)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("exp", type=str, choices=["marsaglia", "beta", "mini_sherpa", "marsaglia_collapsed"])
    parser.add_argument("--arg_file", type=str)
    parser.add_argument("--iw_mode", type=str, required=True,
                        help="Options: prior, ic, biaweights_rootsed, gt, ars-M")
    parser.add_argument("--network_path", type=str, help="Path to a trained network to use for inference")
    parser.add_argument("--num_traces", type=int, default=10000)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_obs", type=int, default=None,
                        help="This argument is only used for the Beta-Bernoulli experiment. Specifies number of times \"heads\" is observed.")
    args = parser.parse_args()

    # Assertions
    if args.exp == "beta" and "SLURM_ARRAY_TASK_ID" in os.environ: # A hack to run beta experiments with fewer job submissions
        assert args.num_obs is None, f"Both num_obs and SLURM_ARRAY_TASK_ID are given ({args.num_obs} and {os.environ['SLRUM_ARRAY_TASK_ID']})"
        args.num_obs = int(os.environ["SLURM_ARRAY_TASK_ID"]) % 30 + 1
    assert args.num_obs is None or args.exp == "beta", f"num_obs argument is only required for the beta experiment."

    set_seed(args.seed)
    if args.arg_file is not None:
        assert args.output_name is None, f"Both of arg_file and output_name cannot be given"
        args.output_name = os.path.splitext(os.path.basename(args.arg_file))[0]
        if args.num_obs is not None:
            args.output_name += f"-{args.num_obs}"
    assert args.output_name is not None, f"Exactly one of arg_file or output_name should be given"
    # Load the argument file
    assert Path(args.arg_file).exists()
    with open(args.arg_file) as f:
        data = DotDict(json.load(f))
        for k,v in data.items():
            assert k not in args.__dict__ or getattr(args, k) is None, f"Conflicting argument: {k}"
            setattr(args, k, v)
    # Update network path
    if args.network_path is not None:
        args.network_path = os.path.join(f"out/{args.exp}/nets", args.network_path)
    print(args)

    log_importance_weights = main(args)

    weights_root = OUTPUTS_ROOT / args.exp / "log_importance_weights"
    out_path = weights_root / args.output_name / args.iw_mode / f"{args.num_traces}_{uuid.uuid4()}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the traces on disk
    np.save(str(out_path), log_importance_weights)
    print(f"Saved {args.num_traces} log imporance weights at {out_path}.")