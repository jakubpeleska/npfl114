import tensorflow as tf
import argparse

from typing import *

from svhn_dataset import SVHN
from bboxes_utils import bboxes_training, bboxes_from_fast_rcnn


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
        image = tf.cast(image, tf.float32) / 255
        classes = tf.cast(classes, tf.float32)
        bboxes = tf.cast(bboxes, tf.float32)
        return image, classes, bboxes

    def image_augmentation(self, image, classes, bboxes):
        image = tf.image.resize_with_crop_or_pad(image, 224, 224)
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

        # dataset = dataset.map(self.image_augmentation)

        if dataset_name != 'test':
            # This is very important! We are padding to same shape per batch!
            dataset = dataset.padded_batch(batch_size=self.batch_size, padding_values=(0.0, 0.0, 1e-8), drop_remainder=True)

            dataset = dataset.map(self.labels_encoding)
        else:
            dataset = dataset.padded_batch(batch_size=self.batch_size, padding_values=(0.0, -1.0, 1e-8))

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
