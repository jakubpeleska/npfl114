#!/usr/bin/env python3
from mnist import MNIST
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
import re
from typing import Tuple
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true",
                    help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=100, type=int,
                    help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.1,
                    type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int,
                    help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

        self._W1 = tf.Variable(
            tf.random.normal([MNIST.W * MNIST.H * MNIST.C,
                             args.hidden_layer], stddev=0.1, seed=args.seed),
            trainable=True,
        )
        self._b1 = tf.Variable(tf.zeros([args.hidden_layer]), trainable=True)

        self._W2 = tf.Variable(tf.random.normal(
            [args.hidden_layer, MNIST.LABELS], stddev=0.1, seed=args.seed), trainable=True)
        self._b2 = tf.Variable(tf.zeros([MNIST.LABELS]), trainable=True)

    def predict(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Define the computation of the network.
        inputs = tf.reshape(inputs, [inputs.shape[0], -1])
        lin1 = inputs @ self._W1 + self._b1
        tanh = tf.nn.tanh(lin1)
        lin2 = tanh @ self._W2 + self._b2
        softmax = tf.nn.softmax(lin2)
        return softmax, lin2, tanh, lin1, inputs

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        for batch in dataset.batches(self._args.batch_size):
            # Compute the input layer, hidden layer and output layer.
            _, lin2_out, lin2_in, lin1_out, lin1_in = self.predict(batch["images"])

            labels = batch["labels"].astype(int)
            labels = tf.one_hot(labels, MNIST.LABELS)
            
            # Start with merged loss and softmax layer
            e_X = tf.math.exp(lin2_out - tf.reduce_max(lin2_out, axis=1, keepdims=True)) # subtract max for better stability
            delta_next = e_X / tf.reduce_sum(e_X, axis=1, keepdims=True) - labels
            
            grad_W2 = (tf.transpose(lin2_in) @ delta_next) / lin2_in.shape[0]
            grad_b2 = tf.reduce_mean(delta_next, axis=0)
            
            delta_next = delta_next @ tf.transpose(self._W2)
            delta_next = delta_next * (1 - lin2_in * lin2_in)
            
            grad_W1 = (tf.transpose(lin1_in) @ delta_next) / lin1_in.shape[0]
            grad_b1 = tf.reduce_mean(delta_next, axis=0)
            
            gradients = [grad_W2, grad_b2, grad_W1, grad_b1]
            variables = [self._W2, self._b2, self._W1, self._b1]

            for variable, gradient in zip(variables, gradients):
                # Perform the SGD update with learning rate `self._args.learning_rate`.
                variable.assign_sub(self._args.learning_rate * gradient)
            

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        # Compute the accuracy of the model prediction
        correct = 0
        for batch in dataset.batches(self._args.batch_size):
            probabilities,_,_,_,_= self.predict(batch["images"])

            # Evaluate how many batch examples were predicted correctly.
            correct += (np.argmax(probabilities, axis=1)
                        == batch["labels"]).sum()

        return correct / dataset.size


def main(args: argparse.Namespace) -> Tuple[float, float]:
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

    # Create the TensorBoard writer
    writer = tf.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        model.train_epoch(mnist.train)

        accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(
            epoch + 1, 100 * accuracy), flush=True)
        with writer.as_default(step=epoch + 1):
            tf.summary.scalar("dev/accuracy", 100 * accuracy)

    test_accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(
        epoch + 1, 100 * test_accuracy), flush=True)
    with writer.as_default(step=epoch + 1):
        tf.summary.scalar("test/accuracy", 100 * accuracy)

    # Return dev and test accuracies for ReCodEx to validate.
    return accuracy, test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
