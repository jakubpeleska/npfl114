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
parser.add_argument("--transformer_dropout", default=0., type=float, help="Transformer dropout.")
parser.add_argument("--transformer_expansion", default=4, type=float, help="Transformer FFN expansion factor.")
parser.add_argument("--transformer_heads", default=4, type=int, help="Transformer heads.")
parser.add_argument("--transformer_layers", default=2, type=int, help="Transformer layers.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    class FFN(tf.keras.layers.Layer):
        def __init__(self, dim, expansion, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.expansion = dim, expansion
            # Create the required layers -- first a ReLU-activated dense
            # layer with `dim * expansion` units, followed by a dense layer
            # with `dim` units without an activation.
            self.hidden = tf.keras.layers.Dense(self.dim * self.expansion, activation='relu')
            self.out = tf.keras.layers.Dense(self.dim)

        def get_config(self):
            return {"dim": self.dim, "expansion": self.expansion}

        def call(self, inputs):
            # Execute the FFN Transformer layer.
            return self.out(self.hidden(inputs))

    class SelfAttention(tf.keras.layers.Layer):
        def __init__(self, dim, heads, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.heads = dim, heads
            self.dim_heads = self.dim // self.heads
            # Create weight matrices W_Q, W_K, W_V, and W_O using `self.add_weight`,
            # each with shape `[dim, dim]`; keep the default for other `add_weight` arguments
            # (which means trainable float32 matrices initialized with `"glorot_uniform"`).
            self.W_Q = self.add_weight("W_Q", [dim, dim])
            self.W_K = self.add_weight("W_K", [dim, dim])
            self.W_V = self.add_weight("W_V", [dim, dim])
            self.W_O = self.add_weight("W_O", [dim, dim])

        def get_config(self):
            return {"dim": self.dim, "heads": self.heads}

        def call(self, inputs: tf.Tensor, mask):
            sh = tf.shape(inputs)
            batch_size, max_sentence_len = sh[0], sh[1]
            # Execute the self-attention layer.
            #
            # Start by computing Q, K and V. In all cases:
            # - first multiply `inputs` by the corresponding weight matrix W_Q/W_K/W_V,
            # - reshape via `tf.reshape` to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - transpose via `tf.transpose` to `[batch_size, heads, max_sentence_len, dim // heads]`.
            # print(inputs.shape, self.W_Q.shape)
            Q = inputs @ self.W_Q
            K = inputs @ self.W_K
            V = inputs @ self.W_V
            
            Q = tf.reshape(Q, [batch_size, max_sentence_len, self.heads, self.dim_heads])
            K = tf.reshape(K, [batch_size, max_sentence_len, self.heads, self.dim_heads])
            V = tf.reshape(V, [batch_size, max_sentence_len, self.heads, self.dim_heads])
            
            Q = tf.transpose(Q, [0,2,1,3])
            K = tf.transpose(K, [0,2,1,3])
            V = tf.transpose(V, [0,2,1,3])

            # Continue by computing the self-attention weights as Q @ K^T,
            # normalizing by the square root of `dim // heads`.
            x = Q @ tf.transpose(K, [0,1,3,2])
            x /= tf.sqrt(tf.cast(self.dim_heads, tf.float32))
            
            # Apply the softmax, but including a suitable mask ignoring all padding words.
            # The original `mask` is a bool matrix of shape `[batch_size, max_sentence_len]`
            # indicating which words are valid (`True`) or padding (`False`).
            # To mask an input to softmax, replace it by -1e9 (theoretically we should use
            # minus infinity, but `tf.math.exp(-1e9)` is also zero because of limited precision).
            mask = tf.cast(tf.cast(mask[:, None], tf.int32) * tf.cast(mask[:, None], tf.int32), tf.bool)
            x = tf.nn.softmax(tf.where(mask[:, None], x, -1e9), axis=3)
            
            # Finally,
            # - take a weighted combination of values V according to the computed attention
            #   (using a suitable matrix multiplication),
            # - transpose the result to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - reshape to `[batch_size, max_sentence_len, dim]`,
            # - multiply the result by the W_O matrix.
            x = x @ V
            x = tf.transpose(x, [0,2,1,3])
            x = tf.reshape(x, [batch_size, max_sentence_len, self.dim])
            return x @ self.W_O


    class PositionalEmbedding(tf.keras.layers.Layer):
        def __init__(self, dim, *args, **kwargs):
            assert dim % 2 == 0  # The `dim` needs to be even to have the same number of sin&cos.
            super().__init__(*args, **kwargs)
            self.dim = dim

        def get_config(self):
            return {"dim": self.dim}

        def call(self, inputs):
            max_sentence_len = tf.shape(inputs)[1]
            # Compute the sinusoidal positional embeddings.
            # They have a shape `[max_sentence_len, self.dim]`, where `self.dim` is even and
            # - for `0 <= i < dim / 2`, the value on index `[pos, i]` should be
            #     `sin(pos / 10_000 ** (2 * i / dim))`
            # - the value on index `[pos, i]` for `i >= dim / 2` should be
            #     `cos(pos / 10_000 ** (2 * (i - dim/2) / dim))`
            # - the `0 <= pos < max_sentence_len` is the sentence index.
            # This order is the same as in the visualization on the slides, but
            # different from the original paper where `sin` and `cos` interleave.
            pos = tf.range(0, max_sentence_len, dtype=tf.float32)
            
            i = tf.range(0, self.dim // 2, dtype=tf.float32)
            part1 = tf.sin(pos[:,None] / 10_000 ** (2 * i[None, :] / self.dim))
            
            j = tf.range(self.dim // 2, self.dim, dtype=tf.float32)
            part2 = tf.cos(pos[:,None] / 10_000 ** (2 * (j[None, :] - self.dim // 2) / self.dim))
            
            return tf.concat([part1, part2], axis=1) 
            

    class Transformer(tf.keras.layers.Layer):
        def __init__(self, layers, dim, expansion, heads, dropout, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layers, self.dim, self.expansion, self.heads, self.dropout = layers, dim, expansion, heads, dropout
            # Create:
            # - the positional embedding layer;
            # - the required number of transformer layers, each consisting of
            #   - a layer normalization and a self-attention layer followed by a dropout layer,
            #   - a layer normalization and a FFN layer followed by a dropout layer.
            self.pe_layer = Model.PositionalEmbedding(self.dim)
            
            self.layer_norm_sa = [tf.keras.layers.LayerNormalization() for _ in range(layers)]
            self.self_attention = [Model.SelfAttention(self.dim, self.heads) for _ in range(layers)]
            self.dropout_sa = [tf.keras.layers.Dropout(self.dropout) for _ in range(layers)]
            self.layer_norm_ffn = [tf.keras.layers.LayerNormalization() for _ in range(layers)]
            self.ffn_layers = [Model.FFN(self.dim, self.expansion) for _ in range(layers)]
            self.dropout_ffn = [tf.keras.layers.Dropout(self.dropout) for _ in range(layers)]

        def get_config(self):
            return {name: getattr(self, name) for name in ["layers", "dim", "expansion", "heads", "dropout"]}

        def call(self, inputs, mask):
            # First compute the positional embeddings.
            pe = self.pe_layer(inputs)

            # Add the positional embeddings to the `inputs` and then
            # perform the given number of transformer layers, composed of
            # - a self-attention sub-layer, followed by
            # - a FFN sub-layer.
            # In each sub-layer, pass the input through LayerNorm, then compute
            # the corresponding operation, apply dropout, and finally add this result
            # to the original sub-layer input. Note that the given `mask` should be
            # passed to the self-attention operation to ignore the padding words.
            inputs += pe
            
            x = inputs
            
            for i in range(self.layers):
                sa = x
                sa = self.layer_norm_sa[i](sa)
                sa = self.self_attention[i](sa, mask)
                sa = self.dropout_sa[i](sa)
                
                x += sa
                
                ffn = x
                ffn = self.layer_norm_ffn[i](ffn)
                ffn = self.ffn_layers[i](ffn)
                ffn = self.dropout_ffn[i](ffn)
                
                x += ffn
                
            return x

    def __init__(self, args, train):
        # Implement a transformer encoder network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        idxs = train.forms.word_mapping(words)

        # Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
        embedded = tf.keras.layers.Embedding(train.forms.word_mapping.vocabulary_size(), args.we_dim)(idxs)

        # Call the Transformer layer:
        # - create a `Model.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        # - when calling the layer, convert the ragged tensor with the input words embedding
        #   to a dense one, and also pass the following argument as a mask:
        #     `mask=tf.sequence_mask(ragged_tensor_with_input_words_embeddings.row_lengths())`
        # - finally, convert the result back to a ragged tensor.
        transformer_layer = Model.Transformer(args.transformer_layers, 
                                              args.we_dim, 
                                              args.transformer_expansion, 
                                              args.transformer_heads, 
                                              args.transformer_dropout)
        row_lengths = embedded.row_lengths()
        mask = tf.sequence_mask(row_lengths)
        transformer_out = transformer_layer(embedded.to_tensor(), mask)
        transformer_out = tf.RaggedTensor.from_tensor(transformer_out, row_lengths)

        # Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that the Dense layer can process
        # a `RaggedTensor` without any problem.
        predictions = tf.keras.layers.Dense(train.tags.word_mapping.vocabulary_size(), activation=tf.nn.softmax)(transformer_out)

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
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the model and train
    model = Model(args, morpho.train)

    # Construct the data for the model, each consisting of the following pair:
    # - a tensor of string words (forms) as input,
    # - a tensor of integer tag ids as targets.
    # To create the tag ids, use the `word_mapping` of `morpho.train.tags`.
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

    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    # Return development and training losses for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items() if "loss" in metric}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
