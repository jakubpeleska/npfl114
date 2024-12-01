#!/usr/bin/env python3
import argparse
from typing import Callable, Tuple
import unittest

import tensorflow as tf

# Bounding boxes and anchors are expected to be Numpy/TensorFlow tensors,
# where the last dimension has size 4.

# For bounding boxes in pixel coordinates, the 4 values correspond to:
TOP: int = 0
LEFT: int = 1
BOTTOM: int = 2
RIGHT: int = 3


def bboxes_area(bboxes: tf.Tensor) -> tf.Tensor:
    """ Compute area of given set of bboxes.

    The computation can be performed either using Numpy or TensorFlow.
    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    If the bboxes.shape is [..., 4], the output shape is bboxes.shape[:-1].
    """
    return tf.maximum(bboxes[..., BOTTOM] - bboxes[..., TOP], 0) \
        * tf.maximum(bboxes[..., RIGHT] - bboxes[..., LEFT], 0)


def bboxes_iou(xs: tf.Tensor, ys: tf.Tensor) -> tf.Tensor:
    """ Compute IoU of corresponding pairs from two sets of bboxes `xs` and `ys`.

    The computation can be performed either using Numpy or TensorFlow.
    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    Note that broadcasting is supported, so passing inputs with
    `xs.shape=[num_xs, 1, 4]` and `ys.shape=[1, num_ys, 4]` produces an output
    with shape `[num_xs, num_ys]`, computing IoU for all pairs of bboxes from
    `xs` and `ys`. Formally, the output shape is `np.broadcast(xs, ys).shape[:-1]`.
    """
    intersections = tf.stack([
        tf.maximum(xs[..., TOP], ys[..., TOP]),
        tf.maximum(xs[..., LEFT], ys[..., LEFT]),
        tf.minimum(xs[..., BOTTOM], ys[..., BOTTOM]),
        tf.minimum(xs[..., RIGHT], ys[..., RIGHT]),
    ], axis=-1)

    xs_area, ys_area, intersections_area = bboxes_area(xs), bboxes_area(ys), bboxes_area(intersections)

    return intersections_area / (xs_area + ys_area - intersections_area)


def get_bbox_size(t: tf.Tensor) -> tf.Tensor:
    return tf.stack([
        t[..., RIGHT] - t[..., LEFT],
        t[..., BOTTOM] - t[..., TOP]
    ], axis=-1)


def get_bbox_center(t: tf.Tensor) -> tf.Tensor:
    return tf.stack([
        (t[..., RIGHT] + t[..., LEFT]) / 2.,
        (t[..., BOTTOM] + t[..., TOP]) / 2.
    ], axis=-1)


def filter_bboxes_in(bboxes: tf.Tensor, classes: tf.Tensor, dim: float) -> Tuple[tf.Tensor, tf.Tensor]:
    dim = tf.cast(dim, tf.float32)
    img = tf.convert_to_tensor([[0., 0., dim, dim]])

    intersections = tf.stack([
        tf.maximum(bboxes[..., TOP], img[..., TOP]),
        tf.maximum(bboxes[..., LEFT], img[..., LEFT]),
        tf.minimum(bboxes[..., BOTTOM], img[..., BOTTOM]),
        tf.minimum(bboxes[..., RIGHT], img[..., RIGHT]),
    ], axis=-1)

    mask = (bboxes_area(intersections) / bboxes_area(bboxes)) >= 0.4
    mask_b = mask[..., None]
    mask_b = tf.broadcast_to(mask_b, (tf.shape(bboxes)[0], 4))

    bboxes = tf.boolean_mask(bboxes, mask_b)
    classes = tf.boolean_mask(classes, mask)
    bboxes = tf.reshape(bboxes, (-1, 4))
    return bboxes, classes


def bbox_to_pascal_voc(t: tf.Tensor) -> tf.Tensor:
    return tf.stack([
        t[..., 1],
        t[..., 0],
        t[..., 3],
        t[..., 2]
    ], axis=-1)


pascal_voc_to_bbox = bbox_to_pascal_voc


def bboxes_resize(bbox: tf.Tensor, rates_h, rates_w, max_h, max_w) -> tf.Tensor:
    return bboxes_clip(tf.stack([
        bbox[..., TOP] * rates_h,
        bbox[..., LEFT] * rates_w,
        bbox[..., BOTTOM] * rates_h,
        bbox[..., RIGHT] * rates_w,
    ], axis=-1), max_h, max_w)


def bboxes_translate(bbox: tf.Tensor, h, w, max_h, max_w) -> tf.Tensor:
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)
    return bboxes_clip(tf.stack([
        bbox[..., TOP] + h,
        bbox[..., LEFT] + w,
        bbox[..., BOTTOM] + h,
        bbox[..., RIGHT] + w,
    ], axis=-1), max_h, max_w)


def bboxes_clip(bbox: tf.Tensor, max_h, max_w) -> tf.Tensor:
    return tf.stack([
        tf.clip_by_value(bbox[..., TOP], 0., tf.cast(max_h, tf.float32)),
        tf.clip_by_value(bbox[..., LEFT], 0., tf.cast(max_w, tf.float32)),
        tf.clip_by_value(bbox[..., BOTTOM], 0., tf.cast(max_h, tf.float32)),
        tf.clip_by_value(bbox[..., RIGHT], 0., tf.cast(max_w, tf.float32)),
    ], axis=-1)


def bboxes_to_fast_rcnn(anchors: tf.Tensor, bboxes: tf.Tensor) -> tf.Tensor:
    """ Convert `bboxes` to a Fast-R-CNN-like representation relative to `anchors`.

    The `anchors` and `bboxes` are arrays of four-tuples (top, left, bottom, right);
    you can use the TOP, LEFT, BOTTOM, RIGHT constants as indices of the
    respective coordinates.

    The resulting representation of a single bbox is a four-tuple with:
    - (bbox_y_center - anchor_y_center) / anchor_height
    - (bbox_x_center - anchor_x_center) / anchor_width
    - log(bbox_height / anchor_height)
    - log(bbox_width / anchor_width)

    If the `anchors.shape` is `[anchors_len, 4]` and `bboxes.shape` is `[anchors_len, 4]`,
    the output shape is `[anchors_len, 4]`.
    """

    bbox_sizes = get_bbox_size(bboxes)
    bbox_centers = get_bbox_center(bboxes)
    anchor_sizes = get_bbox_size(anchors)
    anchor_centers = get_bbox_center(anchors)

    return tf.stack([
        (bbox_centers[..., 1] - anchor_centers[..., 1]) / anchor_sizes[..., 1],
        (bbox_centers[..., 0] - anchor_centers[..., 0]) / anchor_sizes[..., 0],
        tf.math.log(bbox_sizes[..., 1] / anchor_sizes[..., 1]),
        tf.math.log(bbox_sizes[..., 0] / anchor_sizes[..., 0]),
    ], axis=-1)


def bboxes_from_fast_rcnn(anchors: tf.Tensor, fast_rcnns: tf.Tensor) -> tf.Tensor:
    """ Convert Fast-R-CNN-like representation relative to `anchor` to a `bbox`.

    The `anchors.shape` is `[anchors_len, 4]`, `fast_rcnns.shape` is `[anchors_len, 4]`,
    the output shape is `[anchors_len, 4]`.
    """

    anchor_sizes = get_bbox_size(anchors)
    anchor_centers = get_bbox_center(anchors)

    bbox_centers = tf.stack([
        fast_rcnns[..., 1] * anchor_sizes[..., 0] + anchor_centers[..., 0],
        fast_rcnns[..., 0] * anchor_sizes[..., 1] + anchor_centers[..., 1],
    ], axis=-1)

    bbox_sizes = tf.stack([
        tf.exp(fast_rcnns[..., 3]) * anchor_sizes[..., 0],
        tf.exp(fast_rcnns[..., 2]) * anchor_sizes[..., 1],
    ], axis=-1)

    return tf.stack([
        bbox_centers[..., 1] - bbox_sizes[..., 1] / 2.,  # top
        bbox_centers[..., 0] - bbox_sizes[..., 0] / 2.,  # left
        bbox_centers[..., 1] + bbox_sizes[..., 1] / 2.,  # bottom
        bbox_centers[..., 0] + bbox_sizes[..., 0] / 2.,  # right
    ], axis=-1)


N_IMPLICIT_CLASSES = 2
CLS_BACKGROUND = 0
CLS_IGNORE = 1


def bboxes_training(
        anchors: tf.Tensor, gold_classes: tf.Tensor, gold_bboxes: tf.Tensor,
        iou_threshold: float, bg_iou_threshold: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """ Compute training data for object detection.

    Arguments:
    - `anchors` is an array of four-tuples (top, left, bottom, right)
    - `gold_classes` is an array of zero-based classes of the gold objects
    - `gold_bboxes` is an array of four-tuples (top, left, bottom, right)
      of the gold objects
    - `iou_threshold` is a given threshold

    Returns:
    - `anchor_classes` contains for every anchor either 0 for background
      (if no gold object is assigned) or `1 + gold_class` if a gold object
      with `gold_class` is assigned to it
    - `anchor_bboxes` contains for every anchor a four-tuple
      `(center_y, center_x, height, width)` representing the gold bbox of
      a chosen object using parametrization of Fast R-CNN; zeros if no
      gold object was assigned to the anchor

    Algorithm:
    - First, for each gold object, assign it to an anchor with the largest IoU
      (the one with smaller index if there are several). In case several gold
      objects are assigned to a single anchor, use the gold object with smaller
      index.
    - For each unused anchor, find the gold object with the largest IoU
      (again the one with smaller index if there are several), and if the IoU
      is >= iou_threshold, assign the object to the anchor.
    """

    gold_classes = tf.cast(gold_classes, tf.int32)

    anchor_classes = tf.zeros((tf.shape(anchors)[0],), dtype=tf.int32) + CLS_BACKGROUND
    anchor_bboxes = tf.zeros_like(anchors)

    # First, for each gold object, assign it to an anchor with the
    # largest IoU (the one with smaller index if there are several). In case
    # several gold objects are assigned to a single anchor, use the gold object
    # with smaller index.
    ious = bboxes_iou(gold_bboxes[:, tf.newaxis, ...], anchors[tf.newaxis, ...])

    gold_anchor_idxs = tf.argmax(ious, axis=1)
    a_idxs, g_idxs = tf.unique(gold_anchor_idxs)
    g_idxs = tf.math.unsorted_segment_min(tf.range(tf.shape(gold_anchor_idxs)[0], dtype=tf.int32), g_idxs,
                                          tf.shape(a_idxs)[0])
    anchor_classes = tf.tensor_scatter_nd_update(anchor_classes, a_idxs[..., tf.newaxis],
                                                 N_IMPLICIT_CLASSES + tf.gather(gold_classes, g_idxs))
    anchor_bboxes = tf.tensor_scatter_nd_update(anchor_bboxes, a_idxs[..., tf.newaxis], tf.gather(gold_bboxes, g_idxs))

    # For each unused anchor, find the gold object with the largest IoU
    # (again the one with smaller index if there are several), and if the IoU
    # is >= threshold, assign the object to the anchor.
    anchors_unused = tf.where(anchor_classes == CLS_BACKGROUND)
    g = tf.gather(ious, anchors_unused, axis=1)
    if tf.greater(tf.shape(g)[0], 0):
        g_idxs = tf.argmax(g, axis=0)
        mask = tf.gather_nd(ious, tf.concat([g_idxs, anchors_unused], axis=-1)) >= iou_threshold
        a_idxs = tf.boolean_mask(anchors_unused, mask)
        g_idxs = tf.squeeze(tf.boolean_mask(g_idxs, mask), axis=1)
        anchor_classes = tf.tensor_scatter_nd_update(anchor_classes, a_idxs,
                                                     N_IMPLICIT_CLASSES + tf.gather(gold_classes, g_idxs))
        anchor_bboxes = tf.tensor_scatter_nd_update(anchor_bboxes, a_idxs, tf.gather(gold_bboxes, g_idxs))

        anchors_used = tf.where(anchor_classes > CLS_BACKGROUND)
        anchor_bboxes = tf.tensor_scatter_nd_update(
            anchor_bboxes,
            anchors_used,
            bboxes_to_fast_rcnn(tf.gather(anchors, tf.squeeze(anchors_used, axis=1)),
                                tf.gather(anchor_bboxes, tf.squeeze(anchors_used, axis=1)))
        )

    # For all anchors between background_iou_threshold and iou_threshold, give them the 'ignore' indicator.
    max_anchors_iou = tf.reduce_max(ious, axis=0)
    anchors_ignored = tf.logical_and(bg_iou_threshold <= max_anchors_iou, anchor_classes == CLS_BACKGROUND)
    if tf.greater(tf.shape(anchors_ignored)[0], 0):
        anchor_classes = tf.where(anchors_ignored, CLS_IGNORE, anchor_classes)

    return anchor_classes, anchor_bboxes


def main(args: argparse.Namespace) -> Tuple[Callable, Callable, Callable]:
    return bboxes_to_fast_rcnn, bboxes_from_fast_rcnn, bboxes_training


class Tests(unittest.TestCase):
    def test_bboxes_to_from_fast_rcnn(self):
        import numpy as np
        data = [
            [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]],
            [[0, 0, 10, 10], [5, 0, 15, 10], [.5, 0, 0, 0]],
            [[0, 0, 10, 10], [0, 5, 10, 15], [0, .5, 0, 0]],
            [[0, 0, 10, 10], [0, 0, 20, 30], [.5, 1, np.log(2), np.log(3)]],
            [[0, 9, 10, 19], [2, 10, 5, 16], [-0.15, -0.1, -1.20397, -0.51083]],
            [[5, 3, 15, 13], [7, 7, 10, 9], [-0.15, 0, -1.20397, -1.60944]],
            [[7, 6, 17, 16], [9, 10, 12, 13], [-0.15, 0.05, -1.20397, -1.20397]],
            [[5, 6, 15, 16], [7, 7, 10, 10], [-0.15, -0.25, -1.20397, -1.20397]],
            [[6, 3, 16, 13], [8, 5, 12, 8], [-0.1, -0.15, -0.91629, -1.20397]],
            [[5, 2, 15, 12], [9, 6, 12, 8], [0.05, 0, -1.20397, -1.60944]],
            [[2, 10, 12, 20], [6, 11, 8, 17], [0, -0.1, -1.60944, -0.51083]],
            [[10, 9, 20, 19], [12, 13, 17, 16], [-0.05, 0.05, -0.69315, -1.20397]],
            [[6, 7, 16, 17], [10, 11, 12, 14], [0, 0.05, -1.60944, -1.20397]],
            [[2, 2, 12, 12], [3, 5, 8, 8], [-0.15, -0.05, -0.69315, -1.20397]],
        ]
        # First run on individual anchors, and then on all together
        for anchors, bboxes, fast_rcnns in [map(lambda x: [x], row) for row in data] + [zip(*data)]:
            anchors, bboxes, fast_rcnns = [tf.convert_to_tensor(data, tf.float32) for data in
                                           [anchors, bboxes, fast_rcnns]]
            np.testing.assert_almost_equal(bboxes_to_fast_rcnn(anchors, bboxes).numpy(), fast_rcnns.numpy(), decimal=3)
            np.testing.assert_almost_equal(bboxes_from_fast_rcnn(anchors, fast_rcnns).numpy(), bboxes.numpy(),
                                           decimal=3)

    def test_bboxes_training(self):
        import numpy as np
        anchors = tf.convert_to_tensor([[0, 0, 10, 10], [0, 10, 10, 20], [10, 0, 20, 10], [10, 10, 20, 20]], tf.float32)
        for gold_classes, gold_bboxes, anchor_classes, anchor_bboxes, iou in [
            [[1], [[14., 14, 16, 16]], [0, 0, 0, 2], [[0, 0, 0, 0]] * 3 + [[0, 0, np.log(.2), np.log(.2)]], 0.5],
            [[2], [[0., 0, 20, 20]], [3, 0, 0, 0], [[.5, .5, np.log(2), np.log(2)]] + [[0, 0, 0, 0]] * 3, 0.26],
            [[2], [[0., 0, 20, 20]], [3, 3, 3, 3],
             [[y, x, np.log(2), np.log(2)] for y in [.5, -.5] for x in [.5, -.5]], 0.24],
            [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 0, 1],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [-0.35, -0.45, 0.53062, 0.40546]], 0.5],
            [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 2, 1],
             [[0, 0, 0, 0], [0, 0, 0, 0], [-0.1, 0.6, -0.22314, 0.69314], [-0.35, -0.45, 0.53062, 0.40546]], 0.3],
            [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 1, 2, 1],
             [[0, 0, 0, 0], [0.65, -0.45, 0.53062, 0.40546], [-0.1, 0.6, -0.22314, 0.69314],
              [-0.35, -0.45, 0.53062, 0.40546]], 0.17],
        ]:
            gold_classes, anchor_classes = tf.convert_to_tensor(gold_classes, tf.int32), tf.convert_to_tensor(
                anchor_classes, tf.int32)
            gold_bboxes, anchor_bboxes = tf.convert_to_tensor(gold_bboxes, tf.float32), tf.convert_to_tensor(
                anchor_bboxes, tf.float32)
            computed_classes, computed_bboxes = bboxes_training(anchors, gold_classes, gold_bboxes, iou)
            np.testing.assert_almost_equal(computed_classes.numpy(), anchor_classes.numpy(), decimal=3)
            np.testing.assert_almost_equal(computed_bboxes.numpy(), anchor_bboxes.numpy(), decimal=3)


if __name__ == '__main__':
    unittest.main()
