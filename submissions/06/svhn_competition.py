# 8a939b25-3475-4096-a653-2d836d1cbcad
# a0e27015-19ae-4c8f-81bd-bf628505a35a
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
from bboxes_utils import bboxes_from_fast_rcnn, unpack_label, BBOX

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true",
                    help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--evaluate", default=False, action="store_true", help="Do not train, only load trained weights and evaluate test set.")
parser.add_argument("--load_weights", default=False, action="store_true", help="Load pre-trained weights.")
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
    args.ratios = [1.,.5,2.]
    args.scales = [2**(i/3) for i in range(3)]
    n_anchors = len(args.ratios) * len(args.scales)
    dims = 128
    
    dataProcessing = DataProcessing(args)

    if not args.evaluate:
        train = dataProcessing.build_dataset(svhn, "train")
        dev = dataProcessing.build_dataset(svhn, "dev")

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
    
    def loss_fn(y_true, y_pred):
        return cls_loss(y_true, y_pred) + 10 * bbox_loss(y_true, y_pred)
    
    def cls_accuracy(y_true, y_pred):
        y_true = unpack_label(y_true)
        y_pred = unpack_label(y_pred)
        y_true = y_true['classes']
        y_pred = y_pred['classes']
        
        positive_mask = tf.squeeze(tf.greater(y_true, 0.0))
        
        probs = tf.gather(tf.nn.sigmoid(y_pred), tf.cast(y_true, tf.int32) - 1, batch_dims=-1)
        
        probs = tf.where(positive_mask, tf.squeeze(probs), 0.0)
        
        normalizer = tf.reduce_sum(tf.cast(positive_mask, tf.float32), axis=-1)
        accuracy = tf.math.divide_no_nan(tf.reduce_sum(probs, axis=-1), normalizer)
        
        return accuracy
    
    weights_file = "svhn_weights.h5"
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(weights_file, save_best_only=True, save_weights_only=True)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                  loss=loss_fn,
                  metrics=[cls_accuracy]
                  )

    if not args.evaluate:
        if args.load_weights:
            model.load_weights(weights_file, True)
        model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[checkpoint_cb])

    model.load_weights(weights_file, True)
    
    test = dataProcessing.build_test_set(svhn, "dev")

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        for i, batch in test.enumerate():
            print(f'image: {i}', end = '\r')
            images = batch[0]
            img_shape = images.shape[1:3]
            
            label = model.predict_on_batch(images)
            y_pred = unpack_label(label)
            
            bboxes = y_pred["bboxes"]
            anchors = dataProcessing.generate_anchors(img_shape)
            bboxes = bboxes_from_fast_rcnn(anchors, bboxes)
            bboxes = tf.squeeze(bboxes)
            
            scores = tf.nn.sigmoid(y_pred["classes"])
            scores = tf.squeeze(scores)
            bbox_classes = tf.argmax(scores, axis=-1)
            scores = tf.gather(scores, bbox_classes, batch_dims=1)
            
            selected_indices = tf.image.non_max_suppression(bboxes, 
                                                            scores,
                                                            iou_threshold=args.iou_threshold,
                                                            score_threshold=0.15, # TODO: test different values
                                                            max_output_size=5)
            
            predicted_classes = tf.gather(bbox_classes, selected_indices)
            predicted_bboxes = tf.gather(bboxes, selected_indices)
            
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [int(label)] + bbox.numpy().tolist()
            print(*output, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
