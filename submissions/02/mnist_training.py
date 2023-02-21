#!/usr/bin/env python3
import numpy as np
import argparse
import datetime
import os
import re
from typing import Dict
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from mnist import MNIST


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true",
                    help="If given, run functions eagerly.")
parser.add_argument("--decay", default=None,
                    choices=["linear", "exponential", "cosine"], help="Decay type")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=200, type=int,
                    help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01,
                    type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=None,
                    type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float,
                    help="Nesterov momentum to use in SGD.")
parser.add_argument("--optimizer", default="SGD",
                    choices=["SGD", "Adam"], help="Optimizer to use.")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(
            re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
        tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu),
        tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
    ])
    
    model.summary()
    # TODO: maybe wrong
    decay_steps = (mnist.train.size / args.batch_size) * args.epochs

    # Decaying learning rate
    if args.decay is not None:
        if args.decay == "linear":
            learning_rate = tf.optimizers.schedules.PolynomialDecay(args.learning_rate, 
                                                                    decay_steps,
                                                                    end_learning_rate=args.learning_rate_final,
                                                                    power=1.0)
        elif args.decay == "exponential":
            # TODO: maybe wrong
            decay_rate = args.learning_rate_final / args.learning_rate
            learning_rate = tf.optimizers.schedules.ExponentialDecay(args.learning_rate, 
                                                                     decay_steps, 
                                                                     decay_rate)
        elif args.decay == "cosine":
            # TODO: maybe wrong
            alpha = args.learning_rate_final / args.learning_rate
            learning_rate = tf.optimizers.schedules.CosineDecay(args.learning_rate, 
                                                                     decay_steps, 
                                                                     alpha=alpha)
    # Naive learning rate
    else:
        learning_rate = args.learning_rate
    
    
    if args.optimizer == "SGD":
        if args.momentum is not None:
            optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate,
                                                             momentum=args.momentum,
                                                             nesterov=True)
        else:
            optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate)
    
    elif args.optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tb_callback],
    )

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
