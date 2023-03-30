import tensorflow as tf

from pyramid_net import PyramidNet
from head import Head

class RetinaNet(tf.keras.Model):
    def __init__(self, n_classes, n_anchors, **kwargs):
        super().__init__(name="RetinaNet", **kwargs)
        
        self.pyramid = PyramidNet()
        self.cls_head = Head(PyramidNet.FILTERS_OUT, n_classes * n_anchors)
        self.bbox_head = Head(PyramidNet.FILTERS_OUT, n_anchors * 4)
        
    def call(self, inputs, training=False):
        p_out = self.pyramid(inputs, training)
        out = tf.transpose([[self.cls_head(p, training), self.bbox_head(p, training)] for p in p_out], perm=0)
        
        
        
        