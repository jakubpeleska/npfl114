#!/usr/bin/env python3
from typing import Tuple, Dict
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

import bboxes_utils
from svhn_dataset import SVHN

from retina.retina_net import RetinaNet

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=..., type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

def parse_example(example: Dict[str,tf.Tensor]):
    """Do a single image sample preprocessing re-sizing, augmentation, bounding boxes, label encoding, etc.)

    Args:
        example (Dict[tf.Tensor]): 
            Image sample dictionary consisting of image data ("image") and ground truth - bounding boxes("bboxes") and classes("labels").

    Return:
        processed_sample (Tuple[tf.Tensor]): Preprocessed image sample as a tuple of - processed_image, true_cls, true_boxes
    """         
    processed_sample = None
    true_boxes = None
    true_cls = None
    
    print(example["image"].shape)

    return example["image"], example["classes"],  example["bboxes"]

def prepare_data(svhn: SVHN, args: argparse.Namespace):
    
    # def transform_example(example):
    #     anchor_classes, anchor_bboxes = tf.numpy_function(
    #         bboxes_utils.bboxes_training, [anchors, example["classes"], example["bboxes"], 0.5], (tf.int32, tf.float32))
    #     anchor_classes = tf.ensure_shape(anchor_classes, [len(anchors)])
    #     anchor_bboxes = tf.ensure_shape(anchor_bboxes, [len(anchors), 4])
    
    train = svhn.train.map(parse_example)
    dev = svhn.dev.map(parse_example)
    test = svhn.test.map(parse_example)
    
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
    svhn = SVHN()
    
    train, dev, test = prepare_data(svhn, args)
    
    model = RetinaNet(SVHN.LABELS, 9)

    optimizer = 
    model.compile(loss=loss_fn, optimizer=tf.optimizers.Adam())

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        for predicted_classes, predicted_bboxes in ...:
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [label] + list(bbox)
            print(*output, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
