from typing import Literal, Dict, Optional, Any, Tuple

import tensorflow as tf

from bboxes_utils import bboxes_training, bboxes_resize
from svhn_dataset import SVHN
from svhn_anchors import AnchorsFactory


class DatasetFactory:
    def __init__(self, svhn: SVHN,
                 batch_size: int,
                 anchors_factory: AnchorsFactory,
                 iou_threshold: float,
                 bg_iou_threshold: float,
                 match_scales_dim: Optional[int],
                 shuffle_train: bool = True):

        if tf.keras.backend.image_data_format() != 'channels_last':
            raise RuntimeError("Not channels_last!!!")

        self.svhn = svhn
        self.batch_size = batch_size
        self.anchors_factory = anchors_factory
        self.iou_threshold = iou_threshold
        self.bg_iou_threshold = bg_iou_threshold
        self.shuffle_train = shuffle_train
        self.match_scales_dim = match_scales_dim

    def to_dict(self) -> Dict[str, Any]:
        import inspect
        args = list(inspect.signature(self.__class__).parameters.keys())
        return {p: getattr(self, p) for p in args}

    @classmethod
    def from_existing(cls, dataset_factory: 'DatasetFactory', **kwargs) -> 'DatasetFactory':
        in_kwargs = dataset_factory.to_dict()
        in_kwargs.update(kwargs)
        return cls(**in_kwargs)

    def _to_tuple(self, d: dict) -> tuple:
        image, classes, bboxes = d["image"], d["classes"], d["bboxes"]
        return image, {'classes': classes, 'bboxes': bboxes}

    def _fix_image(self, image, *ys):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        return image, *ys

    @staticmethod
    def resize(image, bboxes, new_height: int, new_width: int):
        shape = tf.shape(image)
        h, w = shape[-3], shape[-2]

        maxdim = tf.cond(tf.greater(h, w), lambda: h, lambda: w)
        rate_h = tf.cast(new_height, tf.float32) / tf.cast(maxdim, tf.float32)
        rate_w = tf.cast(new_width, tf.float32) / tf.cast(maxdim, tf.float32)

        h = tf.cast(tf.multiply(tf.cast(h, tf.float32), rate_h), tf.int32)
        w = tf.cast(tf.multiply(tf.cast(w, tf.float32), rate_w), tf.int32)
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

        if bboxes is not None:
            bboxes = bboxes_resize(bboxes, rate_h, rate_w, new_height, new_width)

        return image, bboxes

    @staticmethod
    def encode_dimensions_into_image(x: tf.Tensor, h: int, w: int):
        x = tf.tensor_scatter_nd_update(x,
                                        tf.convert_to_tensor([[0, 0, 0], [0, 0, 1]], tf.int32),
                                        tf.convert_to_tensor(
                                            [tf.cast(h, tf.float32) / 255.0, tf.cast(w, tf.float32) / 255.0],
                                            tf.float32))
        return x

    @staticmethod
    def decode_dimensions_from_images(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        hs = tf.cast(x[..., 0, 0, 0] * 255.0, tf.int32)
        ws = tf.cast(x[..., 0, 0, 1] * 255.0, tf.int32)
        return hs, ws

    def _match_scales(self, image, y):
        shape = tf.shape(image)
        h, w = shape[-3], shape[-2]

        classes = y['classes']
        bboxes = y['bboxes']

        if self.match_scales_dim is not None:
            image, bboxes = self.resize(image, bboxes, self.match_scales_dim, self.match_scales_dim)

        # encode height and width into the original image
        image = self.encode_dimensions_into_image(image, h, w)

        return image, {'classes': classes, 'bboxes': bboxes}

    def _pack_y_true(self, x, y):
        y = tf.concat([
            tf.cast(y['classes'], tf.float32)[..., None],
            y['bboxes'],
        ], axis=-1)
        return x, y

    @staticmethod
    def unpack_y_true(y_true: tf.Tensor) -> Dict[Literal['classes', 'bboxes'], tf.Tensor]:
        classes = tf.cast(y_true[..., 0], tf.int32)
        bboxes = y_true[..., 1:]
        return {'classes': classes, 'bboxes': bboxes}

    def _ragged_to_dense(self, x, y):
        x = x.to_tensor()
        bboxes = y['bboxes']
        classes = y['classes']

        im_shape = tf.shape(x)
        height, width = im_shape[-3], im_shape[-2]
        anchors = self.anchors_factory.build(height=height, width=width)

        def _bboxes_training(x):
            classes, bboxes = x
            classes, bboxes = bboxes_training(anchors=anchors, gold_bboxes=bboxes, gold_classes=classes,
                                              iou_threshold=self.iou_threshold,
                                              bg_iou_threshold=self.bg_iou_threshold)
            return classes, bboxes

        classes, bboxes = tf.map_fn(_bboxes_training, (classes, bboxes), fn_output_signature=(tf.int32, tf.float32))
        return x, {'classes': classes, 'bboxes': bboxes}

    def _fix_bboxes(self, image, classes, bboxes, anchors, iou_threshold: float, bg_iou_threshold: float) -> tuple:
        classes, bboxes = bboxes_training(anchors, classes, bboxes, iou_threshold=iou_threshold,
                                          bg_iou_threshold=bg_iou_threshold)
        return image, classes, bboxes

    def build(self, dataset: Literal['train', 'dev', 'test']):
        out: tf.data.Dataset = getattr(self.svhn, dataset)

        out = out \
            .map(self._to_tuple)

        if dataset == 'train' and self.shuffle_train:
            out = out \
                .shuffle(buffer_size=5000)

        out = out \
            .map(self._fix_image) \
            .map(self._match_scales) \
            .apply(tf.data.experimental.dense_to_ragged_batch(batch_size=self.batch_size)) \
            .map(self._ragged_to_dense)

        return out

    def build_packed(self, dataset: Literal['train', 'dev', 'test']):
        return self.build(dataset) \
            .map(self._pack_y_true)
