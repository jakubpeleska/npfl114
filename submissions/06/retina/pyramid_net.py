import tensorflow as tf


class PyramidNet(tf.keras.Model):
    FILTERS_OUT = 256
    
    def __init__(self, **kwargs):
        super().__init__(name="Pyramid", **kwargs)

        self.resnet = tf.keras.applications.ResNet50(
            include_top=False, input_shape=[None, None, 3]
        )
        self.resnet = tf.keras.Model(
            inputs=[self.resnet.inputs], outputs=[
                self.resnet.get_layer(layer_name).output
                for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
            ]
        )
        
        # self.resnet.summary()
        
        self.c3_to_p3 = tf.keras.layers.Conv2D(self.FILTERS_OUT, 1, 1, padding='same')
        self.c4_to_p4 = tf.keras.layers.Conv2D(self.FILTERS_OUT, 1, 1, padding='same')
        self.c5_to_p5 = tf.keras.layers.Conv2D(self.FILTERS_OUT, 1, 1, padding='same')
        
        # TODO: maybe missing conv for c3,c4,c5
        self.c5_to_p6 = tf.keras.layers.Conv2D(self.FILTERS_OUT, 3, 2, padding='same')
        self.p6_to_p7 = tf.keras.layers.Conv2D(self.FILTERS_OUT, 3, 2, padding='same')
        self.upscale = tf.keras.layers.UpSampling2D(2)

    def call(self, inputs, training=False):
        c3, c4, c5 = self.resnet(inputs, training=training)
        p3 = self.c3_to_p3(c3)
        p4 = self.c4_to_p4(c4)
        p5 = self.c5_to_p5(c5)
        p4 += self.upscale(p5)
        p3 += self.upscale(p4)
        p6 = self.c5_to_p6(c5)
        p7 = self.p6_to_p7(tf.nn.relu(p6))
        
        return p3, p4, p5, p6, p7
