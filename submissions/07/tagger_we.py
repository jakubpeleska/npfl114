#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        # Implement a one-layer RNN network. The input `words` is
        # a `RaggedTensor` of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)
        
        # Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        idxs = train.forms.word_mapping(words)

        # Embed input words with dimensionality `args.we_dim`.
        embedded = tf.keras.layers.Embedding(train.forms.word_mapping.vocabulary_size(), args.we_dim)(idxs)

        # Create the specified `args.rnn` RNN layer ("LSTM" or "GRU") with dimension `args.rnn_dim`.
        rnn_layer = {"LSTM": tf.keras.layers.LSTM, "GRU": tf.keras.layers.GRU}[args.rnn](args.rnn_dim,return_sequences=True)
        rnn_out = tf.keras.layers.Bidirectional(rnn_layer, merge_mode='sum')(embedded)

        # Add a softmax classification layer into as many classes as there are unique tags in the `word_mapping` of `train.tags`.
        predictions = tf.keras.layers.Dense(train.tags.word_mapping.vocabulary_size() ,activation=tf.nn.softmax)(rnn_out)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3

        super().__init__(inputs=words, outputs=predictions)

        def ragged_sparse_categorical_crossentropy(y_true, y_pred):
            return tf.losses.SparseCategoricalCrossentropy()(y_true.values, y_pred.values)

        self.compile(optimizer=tf.optimizers.Adam(jit_compile=False),
                     loss=ragged_sparse_categorical_crossentropy,
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(False)
        tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Construct the data for the model, each consisting of the following pair:
    # - a tensor of string words (forms) as input,
    # - a tensor of integer tag ids as targets.
    def extract_tagging_data(example):
        forms = example["forms"]
        tag_ids = morpho.train.tags.word_mapping(example["tags"])
        return forms, tag_ids

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(extract_tagging_data)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    train, dev = create_dataset("train"), create_dataset("dev")
    
    # Create the model and train
    model = Model(args, morpho.train)

    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    # Return development metrics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)