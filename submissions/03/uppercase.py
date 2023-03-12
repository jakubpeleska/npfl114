#!/usr/bin/env python3
# a0e27015-19ae-4c8f-81bd-bf628505a35a
# 097d9727-7c6f-4127-b1e0-771ebeb3583a

import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `windows`.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=30, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=300, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--dropout_rate", default=0.2, type=float, help="Dropout rate applied on hidden layers.")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--evaluate", default=False, help="Evaluate model.")
parser.add_argument("--hidden_layers", default=2, type=int, help="Number of hidden layers.")
parser.add_argument("--hidden_size", default=750, type=int, help="Size of hidden layers.")
parser.add_argument("--hidden_size_1", default=None, type=int, help="Size of hidden layer 1.")
parser.add_argument("--hidden_size_2", default=None, type=int, help="Size of hidden layer 2.")
parser.add_argument("--model", default="uppercase_model.h5", type=str, help="Output model path.")
parser.add_argument("--prediction", default="uppercase_test.txt", type=str, help="Output test prediction path.")
parser.add_argument("--save_model", default=True, help="Save model.")
parser.add_argument("--save_prediction", default=True, help="Save prediction.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=7, type=int, help="Window size to use.")

def predict(model: tf.keras.Model, dataset: UppercaseData.Dataset):
    to_uppercase = np.argmax(model.predict(dataset.data["windows"]), axis=1)
    text = np.array(list(dataset.text), dtype='unicode')
    for i, upper in enumerate(to_uppercase):
        if bool(upper) and len(text[i].upper()) == 1:
            text[i] = text[i].upper()
        else:    
            text[i] = text[i].lower()
    return "".join(text.tolist())


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

    # Load data
    uppercase_data = UppercaseData(args.window, args.alphabet_size)
     
    input_size = 2 * args.window + 1
    alphabet_size = len(uppercase_data.train.alphabet)
    
    if args.evaluate:
        model = tf.keras.models.load_model(args.model, compile=False)
    else:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=[input_size], dtype=tf.int32))
        model.add(tf.keras.layers.Lambda(lambda x: tf.one_hot(x, alphabet_size)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(args.hidden_size_1 if args.hidden_size_1 is not None else args.hidden_size, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(args.dropout_rate))
        model.add(tf.keras.layers.Dense(args.hidden_size_2 if args.hidden_size_2 is not None else args.hidden_size, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(args.dropout_rate))
        model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
        
        # model.summary()
            
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.losses.SparseCategoricalCrossentropy(),
                    metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")])
        
        # tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)
        
        labels = uppercase_data.train.data["labels"]
        windows = uppercase_data.train.data["windows"]
        keep = np.logical_or(labels == 1, np.random.choice([True,False], p=[1, 0], size=len(labels)))
        
        train_data = {"labels": [], "windows": []}
        
        train_data["labels"] = labels[keep]
        train_data["windows"] = windows[keep]
        
        labels = train_data["labels"]

        # print(len(labels), len(labels[labels == 0]) / len(labels), len(labels[labels == 1])/ len(labels))
        
        model.fit(train_data["windows"], 
                train_data["labels"],
                batch_size=args.batch_size, 
                epochs=args.epochs,
                # callbacks=[tb_callback]
                )
    
    # Generate correctly capitalized test set.
    # os.makedirs(args.logdir, exist_ok=True)
    
    if args.save_model and not args.evaluate:
        # Save the model, without the optimizer state.
        model.save(args.model, include_optimizer=False)
    
    prediction = predict(model, uppercase_data.dev)
    dev_accuracy = uppercase_data.evaluate(uppercase_data.dev, prediction)
    print(f'Dev accuracy: {dev_accuracy}%')
    
    if args.save_prediction:
        # filename = os.path.join(args.logdir, "uppercase_test.txt")
        filename = args.prediction
        prediction = predict(model, uppercase_data.test)
        with open(filename, "w", encoding="utf-8") as predictions_file:
            predictions_file.write(prediction)
            
    return dev_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    
    sizes_1 = [2500]
    dropout_rates = [0.2]
    
    accuracies = np.zeros((len(sizes_1), len(dropout_rates)))
    
    args.epochs = 3
    args.window = 7
    args.alphabet_size = 100
    args.save_model = True
    args.save_prediction = True
    args.hidden_size_2 = 750
    
    for i, size_1 in enumerate(sizes_1):
        for j, dropout_rate in enumerate(dropout_rates):
            args.hidden_size_1 = size_1
            args.dropout_rate = dropout_rate
            accuracies[i, j] = main(args)
            
    print(accuracies)
    print(np.argmax(accuracies))
