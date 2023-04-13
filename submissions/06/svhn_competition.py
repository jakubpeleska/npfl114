#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import *
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from data_processing import DataProcessing
from retina_net import RetinaNet, RetinaNetLoss
from svhn_dataset import SVHN
from bboxes_utils import unpack_label


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true",
                    help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--iou_threshold", default=0.5, type=float,
                    help="Value of intersection of unions to consider the anchor for bbox.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int,
                    help="Maximum number of threads to use.")

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

    # Load the data
    svhn = SVHN()

    args.levels = [5]
    args.ratios = [1.]
    args.scales = [1.]
    n_anchors = len(args.ratios) * len(args.scales)
    dims = 128
    
    dataProcessing = DataProcessing(args)

    train = dataProcessing.build_dataset(svhn, "train")
    dev = dataProcessing.build_dataset(svhn, "dev")
    test = dataProcessing.build_dataset(svhn, "test")

    model = RetinaNet(SVHN.LABELS, n_anchors, dims, args.levels)

    loss = RetinaNetLoss(SVHN.LABELS)
    
    def cls_loss(y_true, y_pred):
        y_pred = unpack_label(y_pred)
        y_pred_cls = y_pred['classes']
        
        return loss.cls_loss(y_true, y_pred_cls)
    
    def bbox_loss(y_true, y_pred):
        y_pred = unpack_label(y_pred)
        y_pred_bbox = y_pred['bboxes']
        
        return loss.bbox_loss(y_true, y_pred_bbox)
    
    def cls_accuracy(y_true, y_pred):
        y_true = unpack_label(y_true)
        y_pred = unpack_label(y_pred)
        y_true = y_true['classes']
        y_pred = y_pred['classes']
        
        positive_mask = tf.greater(y_true, 0.0)
        
        probs = tf.gather(tf.nn.sigmoid(y_pred), tf.cast(y_true, tf.int32) - 1)
        
        probs = tf.boolean_mask(probs, positive_mask)
        
        return tf.reduce_mean(probs, axis=-1)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001, jit_compile=False),
                  loss=[cls_loss, bbox_loss],
                  loss_weights=[1.0, 1.],
                  metrics=[cls_accuracy]
                  )

    model.fit(train, epochs=args.epochs, validation_data=dev)

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
