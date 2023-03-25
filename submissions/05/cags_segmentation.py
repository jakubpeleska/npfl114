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
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=4, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

def prepare_data(cags: CAGS, args: argparse.Namespace):
    train = cags.train.map(lambda example: (example["image"], example["mask"]))
    dev = cags.dev.map(lambda example: (example["image"], example["mask"]))
    test = cags.test.map(lambda example: example["image"])
    
    # Generate batches
    train = train.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    dev = dev.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test = test.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train, dev, test

def upscaling_block(inputs, filters, merge = None):
    x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    if merge is not None:
        x = tf.keras.layers.concatenate([x, merge])
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.activations.relu(x)

def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    cags = CAGS()
    train, dev, test = prepare_data(cags, args)

    # Load the EfficientNetV2-B0 model
    backbone = tf.keras.applications.EfficientNetV2B0(include_top=False)
    backbone = tf.keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(layer).output for layer in [
             "top_activation", "block5e_add", "block3b_add", "block2b_add", "stem_activation"]]
    )
    backbone.trainable = False
    
    # backbone.summary()

    # Create the model and train it
    inputs = tf.keras.Input(shape=[CAGS.H, CAGS.W, CAGS.C])
    backbone_outputs = backbone(inputs, training=False)
    x = upscaling_block(backbone_outputs[0], 112, backbone_outputs[1])
    x = upscaling_block(x, 48, backbone_outputs[2])
    x = upscaling_block(x, 32, backbone_outputs[3])
    x = upscaling_block(x, 32, backbone_outputs[4])
    x = upscaling_block(x, 16)
    outputs = tf.keras.layers.Conv2D(1, 1, padding='same', activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(inputs, outputs)
    
    # model.summary()
    
    model.compile(
            optimizer=tf.optimizers.Adam(jit_compile=False),
            loss=tf.losses.BinaryCrossentropy(),
            metrics=[tf.metrics.BinaryAccuracy(name="accuracy"), tf.metrics.IoU(2, [0,1],name='iou')],
        )
    
    model.fit(train, 
              epochs=args.epochs, 
              validation_data=dev,
              )

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict the masks on the test set
        test_masks = model.predict(test)

        for mask in test_masks:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
