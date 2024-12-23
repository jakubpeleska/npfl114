#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=500, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

def prepare_data(cags: CAGS, args: argparse.Namespace):
    train = cags.train.map(lambda example: (example["image"], example["label"]))
    dev = cags.dev.map(lambda example: (example["image"], example["label"]))
    test = cags.test.map(lambda example: example["image"])
    
    # Generate batches
    train = train.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    dev = dev.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test = test.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train, dev, test


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

    # Load the data
    cags = CAGS()
    
    print(cags.train.element_spec)
    train, dev, test = prepare_data(cags, args)

    # Load the EfficientNetV2-B0 model
    backbone = tf.keras.applications.EfficientNetV2B0(include_top=False, pooling='avg')
    
    backbone.trainable = False

    # Create the model and train it
    inputs = tf.keras.Input(shape=[CAGS.H, CAGS.W, CAGS.C])
    x = backbone(inputs)
    x = tf.keras.layers.Dense(1280, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(CAGS.LABELS), activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs, outputs)
    
    model.summary()
    
    model.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
    
    os.makedirs(args.logdir, exist_ok=True)
    
    checkpoint_path = os.path.join(args.logdir, "model_weights.ckpt")
        
    model.save_weights(checkpoint_path)
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      monitor='val_accuracy',
                                                      mode='max',
                                                      save_weights_only=True,
                                                      save_best_only=True)
    tensorboardb_cb = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)
    
    model.fit(train, 
              epochs=args.epochs, 
              validation_data=dev,
              callbacks=[checkpoint_cb, tensorboardb_cb]
              )
    
    # Load model with best validation accuracy
    model.load_weights(checkpoint_path)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict the probabilities on the test set
        test_probabilities = model.predict(test)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
