# 8a939b25-3475-4096-a653-2d836d1cbcad
# a0e27015-19ae-4c8f-81bd-bf628505a35a
#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import *
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import functools

from homr_dataset import HOMRDataset
from cnn_model import CNNModel

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--cnn_layers", default=4, type=int, help="Number of CNN layers.")
parser.add_argument("--cnn_dim", default=5, type=int, help="CNN layer base dimension as log 2.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--decode", default="greedy", choices=["greedy", "beam"], help="Decode alg.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--height", default=64, type=int, help="Image height.")
parser.add_argument("--evaluate", default=False, action="store_true", help="Do not train, only load trained weights and evaluate test set.")
parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--load_weights", default=None, type=str, help="Load pre-trained weights.")
parser.add_argument("--dropout", default=0.5, type=float, help="Value of dropout for RNN.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=1024, type=int, help="RNN layer dimension.")
parser.add_argument("--rnn_layers", default=5, type=int, help="Number of RNN layers.")
parser.add_argument("--skip_connections", default=False, action="store_true", help="Random seed.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weights_file", default="weights.h5", type=str, help="Name of file for saving the trained weights.")

def process_data(homr: HOMRDataset, args: argparse.Namespace, name: str):
    def parse_example(example):
        image = example['image']
        image = tf.cast(image, dtype=tf.float32) / 255.
        return image, example['marks']
    
    def resize(image, marks, fixed_height):
        shape = tf.shape(image)
        height = shape[0]
        width = shape[1]
        
        ratio = fixed_height / height
        image = tf.image.resize(image, [fixed_height, tf.cast(tf.cast(width, tf.float64) * ratio, tf.int32)])
        image = tf.transpose(image, [1,0,2])
        return image, marks
    
    dataset: tf.data.Dataset = getattr(homr, name)
    dataset = dataset.map(parse_example).map(functools.partial (resize, fixed_height=args.height))
    
    if name == 'train':
        dataset: tf.data.Dataset = dataset.shuffle(5000)
    
    # It allows the pipeline to run in parallel with
    # the training process, dynamically adjusting the buffer size of the
    # prefetched elements.
    dataset = dataset.ragged_batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

class Model(tf.keras.Model):
    RNN_MAPPING = {"LSTM": tf.keras.layers.LSTM, "GRU": tf.keras.layers.GRU}
    
    def get_resnet(self, depth: int, filters_base = 5, residual = 2):
        separator = ','
        max_pool = 'M-2-2'
        parts = []
        for i in range(depth):
            conv_in = f'C-{2**(i+filters_base)}-3-1-same'
            conv_res = [f'CB-{2**(i+filters_base)}-3-1-same' for _ in range(residual)]
            res_block = f'R-[{separator.join(conv_res)}]'
            parts.extend([conv_in, res_block, max_pool])
        return separator.join(parts)
    
    def get_simple_cnn(self, depth: int, filters_base = 5):
        separator = ','
        conv = [f'C-{2**(i+filters_base)}-3-2-same' for i in range(depth)]
        return separator.join(conv)
        
    
    def __init__(self, args: argparse.Namespace) -> None:
        input_shape = [None, args.height, 1]
        inputs: tf.RaggedTensor = tf.keras.layers.Input(shape=input_shape, ragged=True, dtype=tf.float32)
        
        depth = args.cnn_layers
        filters = args.cnn_dim
        architecture = self.get_simple_cnn(depth, filters)
        
        cnn_model = CNNModel(input_shape, None, architecture)
        
        x = inputs.to_tensor(1)
        
        x = cnn_model(x)
        
        shape = tf.shape(x)
        x = tf.reshape(x, [shape[0], shape[1], shape[2]*shape[3]])
        
        for i in range(args.rnn_layers):
            rnn_layer = self.RNN_MAPPING[args.rnn](args.rnn_dim, return_sequences=True)
            _x = tf.keras.layers.Bidirectional(rnn_layer, merge_mode='sum')(x)
            
            if args.dropout > 0.0:
                _x = tf.keras.layers.Dropout(args.dropout)(_x)
            
            if i > 0 and args.skip_connections:
                x += _x
            else:
                x = _x
                
        print(x.shape)
        
        logits = tf.keras.layers.Dense(1 + len(HOMRDataset.MARKS))(x)
        
        logits = tf.RaggedTensor.from_tensor(logits)
        
        super().__init__(inputs=inputs, outputs=logits)

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def ctc_loss(self, gold_labels: tf.RaggedTensor, logits: tf.RaggedTensor) -> tf.Tensor:
        assert isinstance(gold_labels, tf.RaggedTensor), "Gold labels given to CTC loss must be RaggedTensors"
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC loss must be RaggedTensors"

        # Use tf.nn.ctc_loss to compute the CTC loss.
        # - Convert the `gold_labels` to SparseTensor and pass `None` as `label_length`.
        # - Convert `logits` to a dense Tensor and then either transpose the
        #   logits to `[max_audio_length, batch, dim]` or set `logits_time_major=False`
        # - Use `logits.row_lengths()` method to obtain the `logit_length`
        # - Use the last class (the one with the highest index) as the `blank_index`.
        #
        # The `tf.nn.ctc_loss` returns a value for a single batch example, so average
        # them to produce a single value and return it.
        ctc_labels = gold_labels.to_sparse()
        ctc_logits = tf.transpose(logits.to_tensor(), [1,0,2])
        
        loss = tf.nn.ctc_loss(ctc_labels, ctc_logits, 
                              label_length=tf.cast(gold_labels.row_lengths(), tf.int32), logit_length=tf.cast(logits.row_lengths(), tf.int32),
                              blank_index=len(HOMRDataset.MARKS))
        
        loss = tf.clip_by_value(loss, 0.01, 1000)
        return tf.reduce_mean(loss)
        

    def ctc_decode(self, logits: tf.RaggedTensor, method = 'greedy') -> tf.RaggedTensor:
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC predict must be RaggedTensors"

        # Run `tf.nn.ctc_greedy_decoder` or `tf.nn.ctc_beam_search_decoder`
        # to perform prediction.
        # - Convert the `logits` to a dense Tensor and then transpose them
        #   to shape `[max_audio_length, batch, dim]` using `tf.transpose`
        # - Use `logits.row_lengths()` method to obtain the `sequence_length`
        # - Convert the result of the decoded from a SparseTensor to a RaggedTensor
        ctc_logits = tf.transpose(logits.to_tensor(), [1,0,2])
        if method == 'greedy':
            [decoded], _ = tf.nn.ctc_greedy_decoder(ctc_logits, 
                                                sequence_length=tf.cast(logits.row_lengths(), tf.int32), 
                                                blank_index=len(HOMRDataset.MARKS))
        else:
            [decoded], _ = tf.nn.ctc_beam_search_decoder(ctc_logits,
                                                beam_width=25,
                                                sequence_length=tf.cast(logits.row_lengths(), tf.int32))

        predictions = tf.RaggedTensor.from_sparse(decoded)

        assert isinstance(predictions, tf.RaggedTensor), "CTC predictions must be RaggedTensors"
        return predictions

    # We override the `train_step` method, because we do not want to
    # evaluate the training data for performance reasons
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    # We override `predict_step` to run CTC decoding during prediction.
    def predict_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        y_pred = self(data, training=False)
        y_pred = self.ctc_decode(y_pred, 'beam')
        return y_pred

    # We override `test_step` to run CTC decoding during evaluation.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        y_pred = self.ctc_decode(y_pred)
        return self.compute_metrics(x, y, y_pred, None)

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

    # Load the data. The "image" is a grayscale image represented using
    # a single channel of `tf.uint8`s in [0-255] range.
    homr = HOMRDataset()
    
    train, dev = process_data(homr, args, 'train'), process_data(homr, args, 'dev')

    weights_file = args.weights_file
    
    # Create the model and train it
    model = Model(args)
    
    loss_fn = model.ctc_loss
    metrics = [HOMRDataset.EditDistanceMetric()]
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(weights_file, 
                                                       save_best_only=True, 
                                                       save_weights_only=True, 
                                                       monitor='val_edit_distance', 
                                                       mode='min')
    callbacks = [checkpoint_cb, model.tb_callback]
    
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=args.lr), loss=loss_fn, metrics=metrics)

    if not args.evaluate:
        if args.load_weights != None:
            model.load_weights(args.load_weights, True, True)
        model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=callbacks)

    model.load_weights(weights_file, True, True)
    
    test = process_data(homr, args, 'test')

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(test)

        for sequence in predictions:
            print(" ".join(homr.MARKS[mark] for mark in sequence), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
