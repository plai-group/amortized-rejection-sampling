from argparse import ArgumentParser, Namespace
from pathlib import Path
import pickle
import uuid
import json
import os
from pprint import pprint
import pyprob
from pyprob.distributions import Beta

from utils import OUTPUTS_ROOT, DotDict, dispatch_model, set_seed


def get_training_kwargs(args):
    kwargs = {"lstm_dim": args.train.lstm_dim,
              "batch_size": args.train.batch_size,
              "learning_rate_init": args.train.learning_rate}
    if args.exp == "marsaglia" or args.exp == "marsaglia_collapsed":
        kwargs["observe_embeddings"] = {f"obs_{k}": {'dim': 32, 'depth': 1} for k in range(len(args.model.observed_list))}
    elif args.exp == "mini_sherpa":
        kwargs["observe_embeddings"] = {'calorimeter_n_deposits': {'reshape': [1, 20, 20], 'embedding': pyprob.ObserveEmbedding.CNN2D5C}}
    return kwargs


def main(args):
    if args.device is not None:
        pyprob.set_device(args.device)
        print(f"Training on {args.device}")
    # Load the model and initialize observations
    model = dispatch_model(args)
    # Get experiment-specific training kwargs
    kwargs = get_training_kwargs(args)
    # Train the network
    print("-" * 40)
    print("Training IC network:")
    print(f"Number of traces = {args.num_traces}")
    for k, v in kwargs.items():
        print(f"{k} = {v}")
    model.learn_inference_network(num_traces=args.num_traces,
                                  inference_network=pyprob.InferenceNetwork.LSTM,
                                  pre_generate_layers=False,
                                  **kwargs)
    print("-" * 40)
    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("exp", type=str, choices=["marsaglia", "mini_sherpa", "marsaglia_collapsed"])
    parser.add_argument("--arg_file", type=str)
    parser.add_argument("--num_traces", type=int, default=None, help="Number of traces for training the network.")
    parser.add_argument("--output_name", type=str, help="The name to used for the saving the trained checkpoint.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    # Load the argument file
    assert Path(args.arg_file).exists()
    with open(args.arg_file) as f:
        data = DotDict(json.load(f))
        for k,v in data.items():
            assert k not in args.__dict__ or getattr(args, k) is None, f"Conflicting argument: {k}"
            setattr(args, k, v)
    args.num_traces = int(args.num_traces)
    print(args)

    model = main(args)

    nets_root = Path("out/") / args.exp / "nets"
    network_path = nets_root / args.output_name
    network_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the network checkpoint on disk
    model.save_inference_network(str(network_path))
    print(f"Saved the checkpoint at {network_path}.")