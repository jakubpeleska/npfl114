#!/usr/bin/env python3
import argparse
from typing import Tuple

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> Tuple[float, float, float]:
    
    data = dict()
    # Load data distribution, each line containing a datapoint -- a string.
    with open(args.data_path, "r") as f:
        lines = f.readlines()
        entry_value = 1 / len(lines)
        for line in lines:
            point = line.rstrip("\n")
            if data.get(point) is not None:
                data[point] += entry_value
            else:
                data[point] = entry_value

    model = {}
    # Load model distribution, each line `string \t probability`.
    with open(args.model_path, "r") as f:
        for line in f:
            point, prob = line.rstrip("\n").split('\t')
            model[point] = prob

    data_pd = np.zeros(len(data))
    model_pd = np.zeros(len(data))
    for i, (point, prob) in enumerate(data.items()):
        data_pd[i] = prob
        model_pd[i] = model.get(point, 0)
        
    entropy = -(data_pd * np.log(data_pd)).sum()

    cross_entropy = np.inf if len(model_pd[model_pd == 0]) > 0 else -(data_pd * np.log(model_pd)).sum()

    kl_divergence = np.inf if len(model_pd[model_pd == 0]) > 0 else -(data_pd * np.log(model_pd / data_pd)).sum()

    # Return the computed values for ReCodEx to validate.
    return entropy, cross_entropy, kl_divergence


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, cross_entropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(cross_entropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
