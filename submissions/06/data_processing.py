import tensorflow as tf
import argparse

from typing import *

from svhn_dataset import SVHN
from bboxes_utils import bboxes_training, TOP, LEFT, BOTTOM, RIGHT, BBOX

def resize_and_pad_image(
    image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.

    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio


class DataProcessing:
    def __init__(self, args: argparse.Namespace) -> None:
        self.batch_size = args.batch_size
        self.iou_threshold = args.iou_threshold
        self.levels = args.levels
        self.ratios = args.ratios
        self.scales = args.scales

    def generate_anchors(self, img_shape: Tuple[int]):
        anchors = []
        for l in self.levels:
            x_n = tf.math.ceil(img_shape[0] / (2 ** l))
            y_n = tf.math.ceil(img_shape[1] / (2 ** l))
            stride = 2 ** l

            rx = (tf.range(x_n, dtype=tf.float32) + 0.5) * stride
            ry = (tf.range(y_n, dtype=tf.float32) + 0.5) * stride
            centers = tf.stack(tf.meshgrid(rx, ry), axis=-1)
            centers = tf.reshape(tf.tile(centers, [1, 1, 2]), [-1, 4])

            level_anchors = []
            anchor_sqrt = 2 ** (l + 2)

            for scale in self.scales:
                area = (anchor_sqrt * scale) ** 2
                for ratio in self.ratios:
                    h = tf.math.sqrt(area / ratio)
                    w = tf.math.sqrt(area * ratio)
                    anchor_offsets = [-h/2, w/2, h/2, w/2]
                    level_anchors.append(centers + anchor_offsets)

            anchors.append(tf.concat(level_anchors, axis=0))
        return tf.concat(anchors, axis=0)

    def parse_example(self, example: Dict[str, tf.Tensor]):
        image, classes, bboxes = example["image"], example["classes"],  example["bboxes"]
        image = tf.cast(image, tf.float32)
        classes = tf.cast(classes, tf.float32)
        bboxes = tf.cast(bboxes, tf.float32)
        return image, classes, bboxes

    def image_augmentation(self, image, classes, bboxes):
        image, shape, _ = resize_and_pad_image(image, jitter=[256,384])
        bboxes = tf.stack(
            [
                bboxes[:, TOP] * shape[1],
                bboxes[:, LEFT] * shape[0],
                bboxes[:, BOTTOM] * shape[1],
                bboxes[:, RIGHT] * shape[0],
            ],
            axis=-1,
        )
        return image, classes, bboxes

    def labels_encoding(self, batch_images, batch_classes, batch_bboxes):
        img_shape = tf.shape(batch_images)

        anchors = self.generate_anchors(img_shape[1:3])

        labels = []
        for i in range(self.batch_size):
            label = bboxes_training(anchors, batch_classes[i], batch_bboxes[i], self.iou_threshold)
            labels.append(label)

        labels = tf.convert_to_tensor(labels)

        batch_images = tf.keras.applications.efficientnet.preprocess_input(
            batch_images)
        return batch_images, labels

    def build_dataset(self, svhn: SVHN, dataset_name: str):
        dataset: tf.data.Dataset = getattr(svhn, dataset_name)
        dataset = dataset.map(self.parse_example)

        if dataset_name == "train":
            dataset = dataset.shuffle(buffer_size=5000)

        dataset = dataset.map(self.image_augmentation)

        if dataset_name != 'test':
            # This is very important! We are padding to same shape per batch!
            dataset = dataset.padded_batch(batch_size=self.batch_size, padding_values=(0.0, 0.0, 1e-8), drop_remainder=True)

            dataset = dataset.map(self.labels_encoding)
        else:
            dataset = dataset.padded_batch(batch_size=self.batch_size, padding_values=(0.0, 0.0, 1e-8))

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
