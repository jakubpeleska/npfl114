#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

from typing import *

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=500, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--evaluate", default=False, type=bool, help="Evaluate using loaded weights.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--model_weights_checkpoint", default=None, type=str, help="Load saved weights of previously trained model.")
parser.add_argument("--recurrent_dropout", default=0.2, type=float, help="RNN LSTM layer recurrent dropout rate.")
parser.add_argument("--rnn_dim", default=128, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.2, type=float, help="Mask words with the given probability.")

class TaggerModel(tf.keras.Model):
    # A layer setting given rate of elements to zero.
    class MaskElements(tf.keras.layers.Layer):
        def __init__(self, rate: float) -> None:
            super().__init__()
            self._rate = rate

        def get_config(self) -> Dict[str, Any]:
            return {"rate": self._rate}

        def call(self, inputs: tf.RaggedTensor, training: bool) -> tf.RaggedTensor:
            if training:
                mask = tf.random.uniform(inputs.values.shape)
                return tf.RaggedTensor.from_row_limits(tf.where(mask < self._rate, 0, inputs.values), inputs.row_limits())
            else:
                return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        # Implement a one-layer RNN network. The input `words` is
        # a `RaggedTensor` of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        word_ids = train.forms.word_mapping(words)

        # With a probability of `args.word_masking`, replace the input word by an
        # unknown word (which has index 0).
        mask = tf.keras.layers.Dropout(args.word_masking)(tf.ones_like(word_ids, dtype=tf.float32))
        word_ids = word_ids * tf.cast(mask, dtype=tf.int64)

        # word_ids = self.MaskElements(args.word_masking)(word_ids)

        # Embed input words with dimensionality `args.we_dim`.
        embedded_words = tf.keras.layers.Embedding(train.forms.word_mapping.vocabulary_size(), args.we_dim)(word_ids)

        # Create a vector of input words from all batches using `words.values`
        # and pass it through `tf.unique`, obtaining a list of unique words and
        # indices of the original flattened words in the unique word list.
        unique_words, unique_idxs = tf.unique(words.values)

        # Create sequences of letters by passing the unique words through
        # `tf.strings.unicode_split` call; use "UTF-8" as `input_encoding`.
        words_chars = tf.strings.unicode_split(unique_words, input_encoding="UTF-8")

        # Map the letters into ids by using `char_mapping` of `train.forms`
        words_chars_ids = train.forms.char_mapping(words_chars)

        # Embed the input characters with dimensionality `args.cle_dim`.
        embedded_chars = tf.keras.layers.Embedding(train.forms.char_mapping.vocabulary_size(), args.cle_dim)(words_chars_ids)
        
        # Pass the embedded letters through a bidirectional GRU layer
        # with dimensionality `args.cle_dim`, obtaining character-level representations
        # of the whole words, **concatenating** the outputs of the forward and backward RNNs.
        chars_rnn_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.cle_dim))(embedded_chars)

        # Then, convert these character-level word representations into
        # a RaggedTensor of the same shape as `words` using `words.with_values` call.
        chared_words = words.with_values(tf.gather(chars_rnn_out, unique_idxs))

        # Concatenate the word-level embeddings and the computed character-level WEs
        # (in this order).
        concat_words = tf.keras.layers.concatenate([embedded_words, chared_words])

        # Create the RNN layer with dimension `args.rnn_dim`.
        rnn_layer = tf.keras.layers.LSTM(args.rnn_dim, return_sequences=True, recurrent_dropout=args.recurrent_dropout)
        rnn_out = tf.keras.layers.Bidirectional(rnn_layer, merge_mode='sum')(concat_words)

        # Add a softmax classification layer into as many classes as there are unique tags in the `word_mapping` of `train.tags`.
        predictions = tf.keras.layers.Dense(train.tags.word_mapping.vocabulary_size() ,activation=tf.nn.softmax)(rnn_out)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3

        super().__init__(inputs=words, outputs=predictions)

        def ragged_sparse_categorical_crossentropy(y_true, y_pred):
            return tf.losses.SparseCategoricalCrossentropy()(y_true.values, y_pred.values)

        self.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01 ,jit_compile=False),
                     loss=ragged_sparse_categorical_crossentropy,
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


def main(args: argparse.Namespace) -> None:
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
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    # Create the model and train
    model = TaggerModel(args, morpho.train)

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
    
    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")
    
    checkpoint_path = os.path.join(args.logdir, "model_weights.ckpt")
        
    model.save_weights(checkpoint_path)
    
    checkpointCB = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      monitor='val_accuracy',
                                                      mode='max',
                                                      save_weights_only=True,
                                                      save_best_only=True)
    
    if args.model_weights_checkpoint is not None:
        model.load_weights(args.model_weights_checkpoint)

    if not args.evaluate:
        logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback, checkpointCB])
        
        # Load model with best validation accuracy
        model.load_weights(checkpoint_path)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(test)

        tag_strings = morpho.train.tags.word_mapping.get_vocabulary()
        for sentence in predictions:
            for word in np.asarray(sentence):
                print(tag_strings[np.argmax(word)], file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
