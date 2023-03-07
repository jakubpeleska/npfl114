#!/usr/bin/env python3
from mnist import MNIST
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
import re
from typing import Dict
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


parser = argparse.ArgumentParser()

# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
cnn_description = """
Architecture of Convolution Network is described by the cnn argument, which contains comma-separated specifications of the following layers:
    * C-[filters]-[kernel_size]-[stride]-[padding]: Add a convolution layer with ReLU activation and 
        specified number of filters, kernel size, stride and padding. 
    Example: C-10-3-1-same
        
    * CB-[filters]-[kernel_size]-[stride]-[padding]: Same as C-filters-kernel_size-stride-padding, 
        but use batch normalization. In detail, start with a convolution layer without bias and activation,
        then add batch normalization layer, and finally ReLU activation.     
    Example: CB-10-3-1-same
    
    * M-[pool_size]-[stride]: Add max pooling with specified size and stride, using the default "valid" padding. 
    Example: M-3-2
    
    * R-[layers]: Add a residual connection. The layers contain a specification of at least one 
        convolution layer (but not a recursive residual connection R). 
        The input to the R layer should be processed sequentially by layers, 
        and the produced output (after the ReLU non-linearity of the last layer) should
        be added to the input (of this R layer). 
    Example: R-[C-16-3-1-same,C-16-3-1-same]
    
    * F: Flatten inputs. Must appear exactly once in the architecture.
    
    * H-[hidden_layer_size]: Add a dense layer with ReLU activation and specified size. 
    Example: H-100
    
    * D-[dropout_rate]: Apply dropout with the given dropout rate. 
    Example: D-0.5
    
An example architecture might be --cnn=CB-16-5-2-same,M-3-2,F,H-100,D-0.5. You can assume the resulting network is valid; it is fine to crash if it is not.
"""
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
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

        if args.cnn is None:
            raise AssertionError(
                f"No CNN instructions were specified. {cnn_description}")

        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        next_inputs = inputs
        cnn_architecture = self.parse_architecture(args.cnn)

        for layer_type, parameters in cnn_architecture:
            next_inputs = self.apply_layer(next_inputs, layer_type, parameters)

        hidden = next_inputs

        # Add the final output layer
        outputs = tf.keras.layers.Dense(
            MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        if args.logdir is not None:
            self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
            
    def apply_layer(self, inputs: list, layer_type: str, parameters: list):
        if layer_type == 'C':
            filters, kernel_size, stride, padding = parameters
            return tf.keras.layers.Convolution2D(
                int(filters), int(kernel_size), int(stride), padding, activation=tf.nn.relu)(inputs)

        elif layer_type == 'CB':
            filters, kernel_size, stride, padding = parameters
            conv_outputs = tf.keras.layers.Convolution2D(
                int(filters), int(kernel_size), int(stride), padding, use_bias=False)(inputs)
            batch_norm_outputs = tf.keras.layers.BatchNormalization()(conv_outputs)
            return tf.keras.activations.relu(batch_norm_outputs)

        elif layer_type == 'M':
            pool_size, stride = parameters
            return tf.keras.layers.MaxPool2D(
                int(pool_size), int(stride))(inputs)

        elif layer_type == 'R':
            res_block_instructions = parameters[0].replace('[','').replace(']', '')
            res_block_architecture = self.parse_architecture(res_block_instructions)
            next_inputs = inputs
            for res_layer_type, res_parameters in res_block_architecture:
                next_inputs = self.apply_layer(next_inputs, res_layer_type, res_parameters)
            return next_inputs + inputs

        elif layer_type == 'F':
            return tf.keras.layers.Flatten()(inputs)

        elif layer_type == 'H':
            [hidden_layer_size] = parameters
            return tf.keras.layers.Dense(
                int(hidden_layer_size), activation=tf.nn.relu)(inputs)

        elif layer_type == 'D':
            [dropout_rate] = parameters
            return tf.keras.layers.Dropout(
                float(dropout_rate))(inputs)
            
        else:
            raise ValueError(
                f'Unknown layer type "{layer_type}" found! {cnn_description}')
    
    def parse_architecture(self, instructions_str: str):
        instructions = []
        
        instructions_parts = re.split(r",(?!(?:[^,\[\]]+,)*[^,\[\]]+])", instructions_str, 0)

        for ins_part in instructions_parts:
            [layer_type, *parameters] = re.split(r"-(?!(?:[^-\[\]]+-)*[^-\[\]]+])", ins_part, 0)
            instructions.append((layer_type, parameters))

        return instructions


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

    # Create the model and train it
    model = Model(args)

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[model.tb_callback],
    )

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
