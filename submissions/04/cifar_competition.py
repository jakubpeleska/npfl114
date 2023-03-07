#!/usr/bin/env python3
import numpy as np
import argparse
import datetime
import os
import re
from typing import Tuple
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from cnn_model import CNNModel
from cifar10 import CIFAR10
import tensorflow as tf

# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--augment", default=None, type=str, choices=["tf_image", "layers"], help="Augmentation type.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true",help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=1,type=int, help="Number of epochs.")
parser.add_argument("--filters", default=64,type=int, help="Number of filters in convolutional layers.")
parser.add_argument("--save_preditctions", default=True, type=bool,help="Switch to enable saving the predictions on test data.")
parser.add_argument("--show_images", default=False, type=bool,help="Switch to enable showing the augmented images in Tensoboard.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int,help="Maximum number of threads to use.")

def augment_data(cifar: CIFAR10, args: argparse.Namespace):
    train = tf.data.Dataset.from_tensor_slices((cifar.train.data["images"], cifar.train.data["labels"]))
    dev = tf.data.Dataset.from_tensor_slices((cifar.dev.data["images"], cifar.dev.data["labels"]))
    
    # Convert images from tf.uint8 to tf.float32 and scale them to [0, 1] in the process.
    def image_to_float(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.image.convert_image_dtype(image, tf.float32), label

    # Simple data augmentation using `tf.image`.
    generator = tf.random.Generator.from_seed(args.seed)
    def train_augment_tf_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if generator.uniform([]) >= 0.5:
            image = tf.image.flip_left_right(image)
        image = tf.image.resize_with_crop_or_pad(image, CIFAR10.H + 6, CIFAR10.W + 6)
        image = tf.image.resize(image, [generator.uniform([], CIFAR10.H, CIFAR10.H + 12 + 1, dtype=tf.int32),
                                        generator.uniform([], CIFAR10.W, CIFAR10.W + 12 + 1, dtype=tf.int32)])
        image = tf.image.crop_to_bounding_box(
            image, target_height=CIFAR10.H, target_width=CIFAR10.W,
            offset_height=generator.uniform([], maxval=tf.shape(image)[0] - CIFAR10.H + 1, dtype=tf.int32),
            offset_width=generator.uniform([], maxval=tf.shape(image)[1] - CIFAR10.W + 1, dtype=tf.int32),
        )
        return image, label

    # Simple data augmentation using layers.
    def train_augment_layers(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.keras.layers.RandomFlip("horizontal", seed=args.seed)(image)  # Bug, flip always; fixed in TF 2.12.
        image = tf.keras.layers.RandomZoom(0.2, seed=args.seed)(image)
        image = tf.keras.layers.RandomTranslation(0.15, 0.15, seed=args.seed)(image)
        image = tf.keras.layers.RandomRotation(0.1, seed=args.seed)(image)  # Does not always help (too blurry?).
        return image, label
    
    # Use only first 5000 images, shuffle them and change type from tf.uint8 to tf.float32.
    train = train.shuffle(10000, seed=args.seed).map(image_to_float)
    
    # Do the image augmentation
    if args.augment is not None:
        if args.augment == "tf_image":
            train = train.map(train_augment_tf_image)
        elif args.augment == "layers":
            train = train.map(train_augment_layers)
    
    # Generate batches
    train = train.batch(args.batch_size)
    
    # It allows the pipeline to run in parallel with
    # the training process, dynamically adjusting the buffer size of the
    # prefetched elements.
    train = train.prefetch(tf.data.AUTOTUNE)

    if args.show_images:
        summary_writer = tf.summary.create_file_writer(os.path.join(args.logdir, "images"))
        with summary_writer.as_default(step=0):
            for images, _ in train.unbatch().batch(100).take(1):
                images = tf.transpose(tf.reshape(images, [10, 10 * images.shape[1]] + images.shape[2:]), [0, 2, 1, 3])
                images = tf.transpose(tf.reshape(images, [1, 10 * images.shape[1]] + images.shape[2:]), [0, 2, 1, 3])
                tf.summary.image("train/batch", images)
        summary_writer.close()

    # Do not augment the dev dataset
    dev = dev.map(image_to_float).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train, dev


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
        ",".join(("{}={}".format(
            re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    # Load data
    cifar = CIFAR10(size={"dev": 5000})
    
    # Augment data
    train, dev = augment_data(cifar, args)
    
    conv1_filters = 64
    conv1 = f'CB-{conv1_filters}-3-1-same'
    
    conv2_filters = 64
    conv2 = f'R-[CB-{conv2_filters}-3-1-same,CB-{conv2_filters}-3-1-same]'

    conv3_filters = 16
    conv3 = f'CB-{conv3_filters}-3-1-same'
    
    conv4_filters = 16
    conv4 = f'R-[CB-{conv4_filters}-3-1-same,CB-{conv4_filters}-3-1-same]'
    
    max_pool = f'M-2-2,F'
    
    N1 = 512
    N2 = 256
    dropout = 0.5
    lin_class = f'H-{N1},D-{dropout},H-{N2},D-{dropout}'

    # Create model
    model = CNNModel([CIFAR10.H, CIFAR10.W, CIFAR10.C], len(CIFAR10.LABELS),
                     f"{conv1},{conv2},{conv3},{conv4},{max_pool},{lin_class}")
    
    # Learn model
    model.fit(train, epochs=args.epochs, validation_data=dev)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
