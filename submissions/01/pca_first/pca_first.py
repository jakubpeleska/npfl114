#!/usr/bin/env python3
import argparse
import os
from typing import Tuple
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--examples", default=256, type=int, help="MNIST examples to use.")
parser.add_argument("--iterations", default=100, type=int, help="Iterations of the power algorithm.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> Tuple[float, float]:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Load data
    mnist = MNIST()

    data_indices = np.random.choice(mnist.train.size, size=args.examples, replace=False)
    data = tf.convert_to_tensor(mnist.train.data["images"][data_indices])

    data = tf.reshape(data, [data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]])

    # Now compute mean of every feature.
    mean = tf.math.reduce_mean(data, axis=0)

    # Compute the covariance matrix.
    cov = tf.transpose(data - mean) @ (data - mean) / data.shape[0]

    # Compute the total variance, which is the sum of the diagonal
    # of the covariance matrix.
    total_variance = tf.math.reduce_sum(tf.linalg.diag_part(cov))

    # Now run `args.iterations` of the power iteration algorithm.
    v = tf.ones(cov.shape[0], tf.float32)
    for i in range(args.iterations):
        v = tf.linalg.matvec(cov, v)
        s = tf.linalg.norm(v)
        v = v / s

    # The `v` is now approximately the eigenvector of the largest eigenvalue, `s`.
    # We now compute the explained variance, which is the ratio of `s` and `total_variance`.
    explained_variance = s / total_variance

    # Return the total and explained variance for ReCodEx to validate
    return total_variance, 100 * explained_variance


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    total_variance, explained_variance = main(args)
    print("Total variance: {:.2f}".format(total_variance))
    print("Explained variance: {:.2f}%".format(explained_variance))
