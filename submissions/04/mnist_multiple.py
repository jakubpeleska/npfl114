#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true",
                    help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # Create a model with two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        inputs = (tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
                  tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]))

        next_input1 = inputs[0]
        next_input2 = inputs[1]

        conv1_layer = tf.keras.layers.Conv2D(
            10, 3, 2, "valid", activation=tf.nn.relu)
        next_input1 = conv1_layer(next_input1)
        next_input2 = conv1_layer(next_input2)

        conv2_layer = tf.keras.layers.Conv2D(
            20, 3, 2, "valid", activation=tf.nn.relu)
        next_input1 = conv2_layer(next_input1)
        next_input2 = conv2_layer(next_input2)

        flatten = tf.keras.layers.Flatten()
        next_input1 = flatten(next_input1)
        next_input2 = flatten(next_input2)

        dense1_layer = tf.keras.layers.Dense(200, activation=tf.nn.relu)
        next_input1 = dense1_layer(next_input1)
        next_input2 = dense1_layer(next_input2)

        shared_outputs = [next_input1, next_input2]

        next_inputs = tf.keras.layers.Concatenate()(shared_outputs)
        next_inputs = tf.keras.layers.Dense(
            200, activation=tf.nn.relu)(next_inputs)
        direct_comparison = tf.keras.layers.Dense(
            1, activation=tf.nn.sigmoid)(next_inputs)

        digit_classifier = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        digit_1 = digit_classifier(shared_outputs[0])
        digit_2 = digit_classifier(shared_outputs[1])

        indirect_comparison = tf.argmax(
            digit_1, axis=1) > tf.argmax(digit_2, axis=1)

        outputs = {
            "direct_comparison": direct_comparison,
            "digit_1": digit_1,
            "digit_2": digit_2,
            "indirect_comparison": indirect_comparison,
        }

        # Finally, construct the model.
        super().__init__(inputs=inputs, outputs=outputs)

        # Note that for historical reasons, names of a functional model outputs
        # (used for displayed losses/metric names) are derived from the name of
        # the last layer of the corresponding output. Here we instead use
        # the keys of the `outputs` dictionary.
        self.output_names = sorted(outputs.keys())

        self.compile(
            optimizer=tf.keras.optimizers.Adam(jit_compile=False),
            loss={
                "direct_comparison": tf.keras.losses.BinaryCrossentropy(),
                "digit_1": tf.keras.losses.SparseCategoricalCrossentropy(),
                "digit_2": tf.keras.losses.SparseCategoricalCrossentropy(),
            },
            metrics={
                "direct_comparison": [tf.keras.metrics.BinaryAccuracy("accuracy")],
                "indirect_comparison": [tf.keras.metrics.BinaryAccuracy("accuracy")],
            },
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    # Create an appropriate dataset using the MNIST data.
    def create_dataset(
        self, mnist_dataset: MNIST.Dataset, args: argparse.Namespace, training: bool = False
    ) -> tf.data.Dataset:
        # Start by using the original MNIST data
        dataset = tf.data.Dataset.from_tensor_slices(
            (mnist_dataset.data["images"], mnist_dataset.data["labels"]))

        if training:
            dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)

        dataset = dataset.batch(2, drop_remainder=True)

        def create_element(images, labels):
            return (images[0], images[1]), {"direct_comparison": labels[0] > labels[1],
                                            "digit_1": labels[0],
                                            "digit_2": labels[1],
                                            "indirect_comparison": labels[0] > labels[1]}

        dataset = dataset.map(create_element).batch(args.batch_size)

        return dataset


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

    # Load the data
    mnist = MNIST()

    # Create the model
    model = Model(args)

    # Construct suitable datasets from the MNIST data.
    train = model.create_dataset(mnist.train, args, training=True)
    dev = model.create_dataset(mnist.dev, args)

    # Train
    logs = model.fit(train, epochs=args.epochs,
                     validation_data=dev, callbacks=[model.tb_callback])

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
