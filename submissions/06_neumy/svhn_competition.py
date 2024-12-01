#!/usr/bin/env python3
#
#
# 8a939b25-3475-4096-a653-2d836d1cbcad
# a0e27015-19ae-4c8f-81bd-bf628505a35a
#

import argparse
import datetime
import math
import os
import re
from pathlib import Path
from typing import Literal, Dict, Type, Tuple, Any, List, Sequence
from typing import get_args as t_get_args

from svhn_anchors import AnchorsFactory
from svhn_augmentations import DatasetFactory

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import tensorflow as tf

from bboxes_utils import N_IMPLICIT_CLASSES, CLS_IGNORE, bboxes_from_fast_rcnn, bboxes_resize
from svhn_dataset import SVHN

OptimizerType = Literal['adam', 'sgd']

OPTIMIZERS: Dict[OptimizerType, Type[tf.optimizers.Optimizer]] = {
    'adam': tf.optimizers.Adam,
    'sgd': tf.optimizers.SGD
}

BackboneType = Literal['efficientnet', 'resnet']


def eval_float(v: str) -> float:
    try:
        out = eval(v, math.__dict__, {})
    except:
        raise ValueError()

    if type(out) != float and type(out) != int:
        raise TypeError()
    return float(out)


# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", "--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=80, type=int, help="Number of epochs.")
parser.add_argument("--epochs-start", default=None, type=int, help="Number of starter epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--backbone", choices=t_get_args(BackboneType), default="resnet")
parser.add_argument("--augment", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--augment-scale", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--heads", type=int, nargs=2, default=[3, 7])
parser.add_argument("--starting-weights", default=None, type=str)
parser.add_argument("--learning-rate", "--lr", default=3e-4, type=float)
parser.add_argument("--clipnorm", default=None, type=float)
parser.add_argument("--optimizer", default="adam", choices=t_get_args(OptimizerType))
parser.add_argument("--momentum", default=None, type=float)
parser.add_argument("--name", default=None)
parser.add_argument("--alpha", type=float, default=0.25)
parser.add_argument("--gamma", type=float, default=2.0)
parser.add_argument("--label-smoothing", default=0.0, type=float)
parser.add_argument("--iou-threshold", default=0.6, type=float)
parser.add_argument("--iou-threshold-bg", default=0.4, type=float)
parser.add_argument("--anchor-scales", type=eval_float, nargs='+', default=None)
parser.add_argument("--anchor-ratios", type=eval_float, nargs='+', default=None)
parser.add_argument("--bbox-weight", type=float, default=1.)
parser.add_argument("--match-scales-dim", type=int, default=None)
parser.add_argument("--evaluate", type=str, default=None)
parser.add_argument("--evaluate-output", type=str, default="svhn_competition.txt")


class RetinaNet(tf.keras.Model):
    def __init__(self, num_classes: int, backbone: BackboneType, levels: Tuple[int, int], n_anchors: int):
        inp = tf.keras.Input(shape=(None, None, 3))
        if backbone == "efficientnet":
            backbone_a = tf.keras.applications.EfficientNetV2B0(include_top=False, input_shape=(None, None, 3))

            # Extract features of different resolution. Assuming 224x224 input images
            # (you can set this explicitly via `input_shape` of the above constructor),
            # the below model returns five outputs with resolution 7x7, 14x14, 28x28, 56x56, 112x112.
            backbone_m = tf.keras.Model(
                inputs=backbone_a.input,
                outputs=[backbone_a.get_layer(layer).output for layer in [
                    "top_activation", "block5e_add", "block3b_add", "block2b_add", "block1a_project_activation"]],
                name=f"backbone-{backbone}"
            )
        elif backbone == "resnet":
            backbone_a = tf.keras.applications.ResNet50(include_top=False, input_shape=(None, None, 3))
            backbone_m = tf.keras.Model(
                inputs=backbone_a.input,
                outputs=[backbone_a.get_layer(layer_name).output
                         for layer_name in
                         ["conv5_block3_out", "conv4_block6_out", "conv3_block4_out", "conv2_block3_out",
                          "pool1_pool"]],
                name=f"backbone-{backbone}"
            )
        else:
            raise ValueError()

        backbone_m = backbone_m(inp)

        hl, hu = levels
        assert hl <= hu
        assert 1 <= hl
        assert hu <= 7

        cs = {}
        for i in range(hl, min(hu, 5) + 1):
            cs[i] = backbone_m[5 - i]

        ps = {}
        for k, c in cs.items():
            ps[k] = tf.keras.layers.Conv2D(256, 1, 1, "same", name=f"p{k}")(c)

        for i in range(min(hu, 5) - 1, hl - 1, -1):
            p_i_h, p_i_w = tf.shape(ps[i])[1], tf.shape(ps[i])[2]
            p_i1_h, p_i1_w = tf.shape(ps[i + 1])[1], tf.shape(ps[i + 1])[2]
            upsmpl_h = tf.cast(tf.math.ceil(p_i_h / p_i1_h), tf.int32)
            upsmpl_w = tf.cast(tf.math.ceil(p_i_w / p_i1_w), tf.int32)

            ps[i] = tf.keras.layers.Add(name=f"add-p{i}-p{i + 1}")([
                ps[i],
                tf.repeat(
                    tf.repeat(ps[i + 1], upsmpl_h, axis=1, name=f"p{i + 1}-upsmpl-h"),
                    upsmpl_w, axis=2, name=f"p{i + 1}-upsmpl-w"
                )[:, :p_i_h, :p_i_w, :]
            ])

        ps_out = {}
        for i in range(hl, min(hu, 5) + 1):
            ps_out[i] = tf.keras.layers.Conv2D(256, 3, 1, "same", name=f"p_out{i}")(ps[i])

        if 6 <= hu:
            ps_out[6] = tf.keras.layers.Conv2D(256, 3, 2, "same", name=f"p_out{6}")(cs[5])
        if 7 <= hu:
            ps_out[7] = tf.keras.layers.Conv2D(256, 3, 2, "same", name=f"p_out{7}")(tf.keras.layers.ReLU()(ps_out[6]))

        inputs = tf.keras.Input((None, None, 256))

        cls_head = inputs
        for i in range(4):
            cls_head = tf.keras.layers.Conv2D(256, 3, 1, "same", activation="relu", name=f"cls_conv{i}",
                                              kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01))(cls_head)
        cls_head = tf.keras.layers.Conv2D(n_anchors * num_classes, 3, 1, "same", name="cls_out",
                                          kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
                                          bias_initializer=tf.initializers.constant(-math.log((1 - 0.01) / 0.01)))(
            cls_head)
        cls_head = tf.keras.Model(inputs=inputs, outputs=cls_head)

        bbox_head = inputs
        for i in range(4):
            bbox_head = tf.keras.layers.Conv2D(256, 3, 1, "same", activation="relu", name=f"bbox_conv{i}",
                                               kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01))(bbox_head)
        bbox_head = tf.keras.layers.Conv2D(n_anchors * 4, 3, 1, "same", name="bbox_out",
                                           kernel_initializer=tf.initializers.RandomNormal(0.0, 0.01),
                                           bias_initializer='zeros')(bbox_head)
        bbox_head = tf.keras.Model(inputs=inputs, outputs=bbox_head)

        cls_outs = []
        box_outs = []

        for i in range(hl, hu + 1):
            x = ps_out[i]
            h = cls_head(x)
            cls_outs.append(tf.reshape(h, [tf.shape(h)[0], -1, num_classes], name=f"reshape_classes-p{i}"))

            h = bbox_head(x)
            box_outs.append(tf.reshape(h, [tf.shape(h)[0], -1, 4], name=f"reshape_bboxes-p{i}"))

        cls_out = tf.concat(cls_outs, axis=-2, name="concat_classes")
        box_out = tf.concat(box_outs, axis=-2, name="concat_bboxes")

        outs = {'classes': cls_out, 'bboxes': box_out}

        super().__init__(inputs=inp, outputs=outs)
        self.backbone = backbone_a
        self.levels = levels


class RetinaNetInference(tf.keras.Model):
    def __init__(self,
                 model: RetinaNet,
                 anchors_factory: AnchorsFactory,
                 max_output_size_per_class: int,
                 max_total_size: int,
                 iou_threshold: float = 0.5,
                 any_iou_threshold: float = 0.7,
                 score_threshold: float = float('-inf'),
                 ):
        super().__init__()
        self.model = model
        self.anchors_factory = anchors_factory
        self.max_output_size_per_class = max_output_size_per_class
        self.max_total_size = max_total_size
        self.iou_threshold = iou_threshold
        self.any_iou_threshold = any_iou_threshold
        self.score_threshold = score_threshold

    def to_dict(self) -> Dict[str, Any]:
        import inspect
        args = list(inspect.signature(self.__class__).parameters.keys())
        return {p: getattr(self, p) for p in args}

    @classmethod
    def from_existing(cls, imodel: 'RetinaNetInference', **kwargs) -> 'RetinaNetInference':
        in_kwargs = imodel.to_dict()
        in_kwargs.update(kwargs)
        return cls(**in_kwargs)

    def call(self, x):
        y = self.model(x)
        y_cls = y['classes']
        y_bbox = y['bboxes']

        h, w = tf.shape(x)[-3], tf.shape(x)[-2]
        anchors = self.anchors_factory.build(h, w)

        bboxes = bboxes_from_fast_rcnn(anchors, y_bbox)[..., tf.newaxis, :]
        scores = tf.nn.sigmoid(y_cls)

        bboxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=bboxes,
            scores=scores,
            max_output_size_per_class=self.max_output_size_per_class,
            max_total_size=self.max_total_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            clip_boxes=False
        )

        # resize back
        orig_hs, orig_ws = DatasetFactory.decode_dimensions_from_images(x)
        rates_h = tf.cast(orig_hs, tf.float32) / tf.cast(h, tf.float32)
        rates_w = tf.cast(orig_ws, tf.float32) / tf.cast(w, tf.float32)

        bboxes = bboxes_resize(bboxes, rates_h[..., None], rates_w[..., None], orig_hs[..., None], orig_ws[..., None])

        idx, num_valid = tf.image.non_max_suppression_padded(bboxes, scores,
                                                             pad_to_max_output_size=True,
                                                             max_output_size=self.max_total_size,
                                                             iou_threshold=self.any_iou_threshold)
        bboxes = tf.RaggedTensor.from_tensor(bboxes, lengths=valid_detections)
        scores = tf.RaggedTensor.from_tensor(scores, lengths=valid_detections)
        classes = tf.RaggedTensor.from_tensor(classes, lengths=valid_detections)

        idx = tf.RaggedTensor.from_tensor(idx, lengths=num_valid)
        idx_flat = (tf.cast(idx, tf.int64) + bboxes.row_starts()[..., None]).values
        bboxes = tf.RaggedTensor.from_row_lengths(tf.gather(bboxes.values, idx_flat), idx.row_lengths())
        scores = tf.RaggedTensor.from_row_lengths(tf.gather(scores.values, idx_flat), idx.row_lengths())
        classes = tf.RaggedTensor.from_row_lengths(tf.gather(classes.values, idx_flat), idx.row_lengths())

        return {
            'bboxes': bboxes,
            'scores': scores,
            'classes': classes,
        }

    def to_python(self, y_pred: Dict[Literal['classes', 'bboxes'], tf.Tensor]) \
            -> Sequence[Tuple[List[int], List[SVHN.BBox]]]:
        classes = tf.cast(y_pred['classes'], tf.int32)
        bboxes = y_pred['bboxes']

        classes = classes.to_list() if isinstance(classes, tf.RaggedTensor) else tf.unstack(classes, axis=0)
        bboxes = bboxes.to_list() if isinstance(bboxes, tf.RaggedTensor) else tf.unstack(bboxes, axis=0)

        return list(zip(classes, bboxes))


def create_retinanet_binary_focal_crossentropy(alpha=0.25, gamma=2.0, y_true_packed=True):
    def retinanet_binary_focal_crossentropy(y_true, y_pred_cls):
        if y_true_packed:
            y_true = DatasetFactory.unpack_y_true(y_true)
        y_true_cls = y_true['classes']
        nonignore_mask = tf.logical_or(y_true_cls < CLS_IGNORE, y_true_cls > CLS_IGNORE)
        positive_mask = (y_true_cls >= N_IMPLICIT_CLASSES)
        y_true_one_hot = tf.one_hot(y_true_cls, depth=SVHN.LABELS + N_IMPLICIT_CLASSES)[..., N_IMPLICIT_CLASSES:]
        y_true_one_hot = tf.where(nonignore_mask[..., None], y_true_one_hot, 0.)
        y_pred_cls = tf.where(nonignore_mask[..., None], y_pred_cls, 0.)

        # number of positive anchors per batch sample
        positive_count = tf.reduce_sum(tf.cast(positive_mask, tf.float32), axis=-1)

        val = tf.reduce_sum(
            tf.keras.backend.binary_focal_crossentropy(y_true_one_hot, y_pred_cls,
                                                       from_logits=True,
                                                       apply_class_balancing=True,
                                                       alpha=alpha,
                                                       gamma=gamma),
            axis=-1
        )  # sum over classes

        val = tf.reduce_sum(val, axis=-1)  # sum over anchors

        return tf.math.divide_no_nan(val, positive_count)

    return retinanet_binary_focal_crossentropy


def create_retinanet_bbox_huber(y_true_packed=True):
    def retinanet_bbox_huber(y_true, y_pred_bbox):
        if y_true_packed:
            y_true = DatasetFactory.unpack_y_true(y_true)
        y_true_bbox = y_true['bboxes']
        y_true_cls = y_true['classes']

        positive_mask = (y_true_cls >= N_IMPLICIT_CLASSES)

        y_true_bbox = tf.where(positive_mask[..., None], y_true_bbox, 0.)
        y_pred_bbox = tf.where(positive_mask[..., None], y_pred_bbox, 0.)

        error = tf.subtract(y_true_bbox, y_pred_bbox)
        abs_error = tf.abs(error)
        half = tf.convert_to_tensor(0.5, dtype=abs_error.dtype)
        delta = tf.convert_to_tensor(1.0, dtype=abs_error.dtype)

        # number of positive anchors per batch sample
        positive_count = tf.reduce_sum(tf.cast(positive_mask, tf.float32), axis=-1)

        val = tf.reduce_sum(
            tf.where(
                abs_error <= delta,
                half * tf.square(error),
                delta * abs_error - half * tf.square(delta),
            ),
            axis=-1
        )  # sum over the 4 bounding box values

        val = tf.reduce_sum(val, axis=-1)  # sum over anchors

        return tf.math.divide_no_nan(val, positive_count)

    return retinanet_bbox_huber


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

    anchors_factory = AnchorsFactory(levels=args.heads, scales=args.anchor_scales, ratios=args.anchor_ratios)
    dataset_factory = DatasetFactory(svhn=svhn, batch_size=args.batch_size, anchors_factory=anchors_factory,
                                     iou_threshold=args.iou_threshold, bg_iou_threshold=args.iou_threshold_bg,
                                     match_scales_dim=args.match_scales_dim, shuffle_train=True)

    model = RetinaNet(num_classes=svhn.LABELS, backbone=args.backbone, levels=args.heads,
                      n_anchors=anchors_factory.anchors_per_unit)
    train_dataset = dataset_factory.build_packed('train')
    dev_dataset = dataset_factory.build_packed('dev')

    # Train the model
    if args.evaluate is None:
        if args.starting_weights is not None:
            model.load_weights(args.starting_weights)

        def _get_optimizer():
            optimizer_kwargs = {}
            if args.clipnorm is not None:
                optimizer_kwargs['clipnorm'] = args.clipnorm
            if args.momentum is not None:
                optimizer_kwargs['momentum'] = args.momentum

            return OPTIMIZERS[args.optimizer](learning_rate=args.learning_rate, **optimizer_kwargs)

        Path(args.logdir).mkdir(exist_ok=True, parents=True)

        callbacks = []
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=args.logdir))
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                monitor='val_loss', filepath=str(Path(args.logdir) / 'weights'),
                save_weights_only=True, save_best_only=True, mode='min',
                verbose=1
            )
        )

        bbox_loss = create_retinanet_bbox_huber()
        cls_loss = create_retinanet_binary_focal_crossentropy(alpha=args.alpha, gamma=args.gamma)

        losses = {'bboxes': bbox_loss, 'classes': cls_loss}
        loss_weights = {'bboxes': args.bbox_weight, 'classes': 1.}

        if args.epochs_start is None:
            args.epochs_start = args.epochs

        model.backbone.trainable = False
        model.compile(optimizer=_get_optimizer(), loss=losses, loss_weights=loss_weights)
        model.fit(train_dataset, epochs=args.epochs_start, callbacks=callbacks, validation_data=dev_dataset)

        if args.epochs_start < args.epochs:
            model.backbone.trainable = True
            model.compile(optimizer=_get_optimizer(), loss=losses, loss_weights=loss_weights)
            model.fit(train_dataset, epochs=args.epochs, callbacks=callbacks, validation_data=dev_dataset,
                      initial_epoch=args.epochs_start)
    else:
        model.load_weights(args.evaluate)
        imodel = RetinaNetInference(model, anchors_factory,
                                    max_output_size_per_class=10,
                                    max_total_size=10,
                                    iou_threshold=0.2,
                                    any_iou_threshold=0.4,
                                    score_threshold=0.4)
        test_dataset = dataset_factory.build('test')
        y_pred = imodel.to_python(imodel.predict(test_dataset))
        with open(args.evaluate_output, "w", encoding="utf-8") as predictions_file:
            for predicted_classes, predicted_bboxes in y_pred:
                output = []
                for label, bbox in zip(predicted_classes, predicted_bboxes):
                    output += [label] + list(bbox)
                print(*output, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
