from pathlib import Path


OUTPUTS_ROOT = Path("../out")


def dispatch_model(args):
    """Initializes a model according to the given arguments

    Args:
        args (argparse.Namespace): a namespace with "exp" and "model" attributes.
        exp is a string and model is a dictionary.
    """
    if args.exp == "marsaglia":
        from exp_marsaglia.model import GUMMarsaglia
        model = GUMMarsaglia(prior_mean=args.model.prior_mean,
                             prior_std=args.model.prior_std,
                             likelihood_std=args.model.likelihood_std,
                             num_obs=len(args.model.observed_list))
    elif args.exp == "marsaglia_collapsed":
        from exp_marsaglia.model import GUMMarsagliaCollapsed
        model = GUMMarsagliaCollapsed(prior_mean=args.model.prior_mean,
                                      prior_std=args.model.prior_std,
                                      likelihood_std=args.model.likelihood_std,
                                      num_obs=len(args.model.observed_list))
    elif args.exp == "beta":
        from exp_beta.model import RejectionBetaBernoulli
        model = RejectionBetaBernoulli(beta=args.model.beta, num_obs=args.num_obs)
    elif args.exp == "mini_sherpa":
        from exp_mini_sherpa.model import MiniSherpa
        model = MiniSherpa()
    else:
        raise ValueError(f"Unexpected argument exp: {args.exp}")
    return model

class DotDict(dict):
    """
    https://stackoverflow.com/questions/13520421/recursive-dotdict/13520518
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            elif isinstance(value, list):
                value = [x for x in value]
            self[key] = value


class TraceWeightModeError(Exception):
    pass


def set_seed(seed):
    if seed is not None:
        import random
        import numpy as np
        import torch
        import pyprob
        np.random.seed(seed)
        random.seed(seed)
        pyprob.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # https://pytorch.org/docs/stable/notes/randomness.html