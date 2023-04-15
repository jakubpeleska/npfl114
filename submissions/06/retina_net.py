import tensorflow as tf

from bboxes_utils import BBOX

from typing import *

class RetinaNetLoss:
    class RetinaNetBBoxLoss(tf.losses.Loss):
        """Implements Smooth L1 loss"""

        def __init__(self, delta, **kwargs):
            super().__init__(**kwargs, name="RetinaNetBBoxLoss")
            self.delta = delta
            
        def call(self, y_true, y_pred):
            positive_mask = tf.greater(y_true[..., BBOX], 0.0)
            y_true = y_true[..., :BBOX]

            loss = tf.keras.losses.huber(y_true, y_pred, self.delta)
            
            loss = tf.boolean_mask(loss, positive_mask)
            return loss
            # normalizer = tf.reduce_sum(tf.cast(positive_mask, tf.int8), axis=-1)
            # return tf.math.divide_no_nan(loss, normalizer)
        
        
    class RetinaNetClassificationLoss(tf.losses.Loss):
        """Implements Focal loss"""

        def __init__(self, n_classes, alpha, gamma, **kwargs):
            super().__init__(**kwargs, name="RetinaNetClassificationLoss")
            self.n_classes = n_classes
            self.alpha = alpha
            self.gamma = gamma
        
        def call(self, y_true, y_pred):
            positive_mask = tf.greater(y_true[..., BBOX], 0.0)
            non_ignore_mask = tf.greater(y_true[..., BBOX], -1.0)
            
            y_true = tf.one_hot(
                tf.cast(y_true[..., BBOX] - 1, dtype=tf.int32),
                depth=self.n_classes,
                dtype=tf.float32,
            )
            y_pred = tf.cast(y_pred, tf.float32)
            
            loss = tf.keras.backend.binary_focal_crossentropy(y_true, y_pred, apply_class_balancing=True,
                                                             alpha=self.alpha, gamma=self.gamma, 
                                                             from_logits=True, )
            
            loss = tf.reduce_sum(loss, axis=-1)
            loss = tf.where(non_ignore_mask, loss, 0.0)
            loss = tf.where(positive_mask, loss*10, loss)
            loss = tf.reduce_sum(loss, axis=-1)
            normalizer = tf.reduce_sum(tf.cast(non_ignore_mask, tf.float32), axis=-1)
            return tf.math.divide_no_nan(loss, normalizer)
    
    def __init__(self, n_classes, alpha = 0.25, gamma = 2.0, delta = 1.0):
        self.cls_loss = self.RetinaNetClassificationLoss(n_classes, alpha, gamma)
        self.bbox_loss = self.RetinaNetBBoxLoss(delta)

class Head(tf.keras.Model):
    def __init__(self, in_features: int, out_features: int, bias_init: Union[tf.keras.initializers.Initializer, str] = "zeros"):
        kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
        inputs = tf.keras.Input((None, None, in_features))
        x = inputs
        for _ in range(4):
            x = tf.keras.layers.Conv2D(
                in_features, 3, padding="same", activation="relu", kernel_initializer=kernel_init)(x)
        out = tf.keras.layers.Conv2D(
            out_features, 3, 1, "same", kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

        super().__init__(inputs=inputs, outputs=out)


class RetinaNet(tf.keras.Model):
    def __init__(self, n_classes: int, n_anchors: int, dims: int = 256, levels: List[int] = [3, 4, 5, 6, 7]) -> None:

        # Load the EfficientNetV2-B0 model. It assumes the input images are
        # represented in [0-255] range using either `tf.uint8` or `tf.float32` type.
        backbone = tf.keras.applications.EfficientNetV2B0(include_top=False)

        # Extract features of different resolution. Assuming 224x224 input images
        # (you can set this explicitly via `input_shape` of the above constructor),
        # the below model returns five outputs with resolution 28x28, 14x14, 7x7.
        backbone = tf.keras.Model(
            inputs=backbone.input,
            outputs=[backbone.get_layer(layer).output for layer in [
                "block3b_add", "block5e_add", "top_activation"]]
        )

        images = tf.keras.layers.Input(shape=[None, None, 3], dtype=tf.float32)

        scale_up = tf.keras.layers.UpSampling2D(2)

        tf.keras.layers.Conv2D(dims, 1, 1, "same")

        c3, c4, c5 = backbone(images)

        p3 = tf.keras.layers.Conv2D(dims, 1, 1, "same")(c3)
        p4 = tf.keras.layers.Conv2D(dims, 1, 1, "same")(c4)
        p5 = tf.keras.layers.Conv2D(dims, 1, 1, "same")(c5)
        p4 = p4 + scale_up(p5)
        p3 = p3 + scale_up(p4)
        p3 = tf.keras.layers.Conv2D(dims, 3, 1, "same")(p3)
        p4 = tf.keras.layers.Conv2D(dims, 3, 1, "same")(p4)
        p5 = tf.keras.layers.Conv2D(dims, 3, 1, "same")(p5)
        p6 = tf.keras.layers.Conv2D(dims, 3, 2, "same")(c5)
        p7 = tf.keras.layers.Conv2D(dims, 3, 2, "same")(tf.nn.relu(p6))

        features = [p3, p4, p5, p6, p7]
        _features = []
        for i, f in enumerate(features):
            if levels.__contains__(i+3):
                _features.append(f)

        features = _features

        prior_probability = tf.initializers.constant(
            -tf.math.log((1 - 0.01) / 0.01))
        cls_head = Head(dims, n_anchors * n_classes,
                        bias_init=prior_probability)
        bbox_head = Head(dims, n_anchors * BBOX, bias_init="zeros")

        cls_outs, bbox_outs = [], []
        for f in features:
            cls_out = cls_head(f)
            cls_outs.append(tf.reshape(
                cls_out, [tf.shape(cls_out)[0], -1, n_classes]))

            bbox_out = bbox_head(f)
            bbox_outs.append(tf.reshape(
                bbox_out, [tf.shape(bbox_out)[0], -1, BBOX]))

        cls_out = tf.concat(cls_outs, axis=-2)
        bbox_out = tf.concat(bbox_outs, axis=-2)
        
        out = tf.concat([bbox_out, cls_out], axis=-1)

        super().__init__(inputs=images, outputs=out)
