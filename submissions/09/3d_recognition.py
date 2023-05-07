# 8a939b25-3475-4096-a653-2d836d1cbcad
# a0e27015-19ae-4c8f-81bd-bf628505a35a
#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

import functools
from scipy import ndimage
import random

from modelnet import ModelNet
from cnn_model import CNNModel

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--evaluate", default=False, action="store_true", help="Evaluate model only.")
parser.add_argument("--epochs", default=25, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weights_file", default="weights.h5", type=str, help="Name of file for saving the trained weights.")

@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        axes = [(0,1), (1,2), (0,2)]
        # pick angles at random
        angle = tf.random.uniform([1], -45, 45)
        # rotate volume
        volume = ndimage.rotate(volume, angle[0], axes=random.choice(axes), reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume

def augmentation(volume: tf.Tensor, label):
    volume = tf.cast(volume, tf.float32)
    volume = rotate(volume)
    return volume, label

def process_data(modelnet: ModelNet, args: argparse.Namespace):
    train = tf.data.Dataset.from_tensor_slices((modelnet.train.data["voxels"], modelnet.train.data["labels"]))
    dev = tf.data.Dataset.from_tensor_slices((modelnet.dev.data["voxels"], modelnet.dev.data["labels"]))
    test = tf.data.Dataset.from_tensor_slices(modelnet.test.data["voxels"])
    
    train = train.map(augmentation)
    
    # Use only the first 5000 images, shuffle them and change type from tf.uint8 to tf.float32.
    train = train.shuffle(5000, seed=args.seed)
    
    # It allows the pipeline to run in parallel with
    # the training process, dynamically adjusting the buffer size of the
    # prefetched elements.
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
        # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    modelnet = ModelNet(args.modelnet)
    
    train, dev, test = process_data(modelnet, args)

    # Model architecture scale
    scale = 1
    
    # VGG16-like model
    conv1_filters = int(32*scale)
    conv1_repeat = 1
    conv1 = f'CB-{conv1_filters}-3-1-same,' * conv1_repeat
    max_pool1 = f'M-2-2'
    
    block1 = f'{conv1}{max_pool1}'
    
    conv2_filters = int(64*scale)
    conv2_repeat = 1
    conv2 = f'CB-{conv2_filters}-3-1-same,' * conv2_repeat
    max_pool2 = f'M-2-2'
    
    block2 = f'{conv2}{max_pool2}'
    
    conv3_filters = int(128*scale)
    conv3_repeat = 2
    conv3 = f'CB-{conv3_filters}-3-1-same,' * conv3_repeat
    max_pool3 = f'M-2-2'
    
    block3 = f'{conv3}{max_pool3}'

    conv4_filters = int(256*scale)
    conv4_repeat = 2
    conv4 = f'CB-{conv4_filters}-3-1-same,' * conv4_repeat
    max_pool4 = f'M-2-2'
    
    block4 = f'{conv4}{max_pool4}'
    
    N1 = int(256*scale)
    N2 = int(128*scale)
    dropout = 0.5
    lin_class = f'F,H-{N1},D-{dropout},H-{N2},D-{dropout}'
    
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)

    # Create model
    model = CNNModel([modelnet.D, modelnet.H, modelnet.W, modelnet.C], len(modelnet.LABELS),
                     f"{block1},{block2},{block3},{block4},{lin_class}", 
                     dim=3, logdir=args.logdir)
    
    model.compile(
            optimizer=tf.optimizers.Adam(0.00001),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
    
    weights_file = args.weights_file
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=weights_file,
                                                      monitor='val_accuracy',
                                                      mode='max',
                                                      save_weights_only=True,
                                                      save_best_only=True)
    
    if not args.evaluate:
        model.load_weights(weights_file)
        # Learn model
        model.fit(train, 
                epochs=args.epochs, 
                validation_data=dev,
                callbacks=[model.tb_callback, checkpoint_cb]
                )
    
    model.load_weights(weights_file)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the probabilities on the test set
        test_probabilities = model.predict(test)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
