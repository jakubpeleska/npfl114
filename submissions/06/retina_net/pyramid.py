import tensorflow as tf

class Pyramid(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(name="Pyramid", **kwargs)
        
        # TODO: we might want to change this
        backbone = tf.keras.applications.ResNet50(
            include_top=False, input_shape=[None, None, 3]
        )
        c3_output, c4_output, c5_output = [
            backbone.get_layer(layer_name).output
            for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
        ]
        backbone = tf.keras.Model(
            inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
        )
        # TODO: add layers
    
    
    def call(self, inputs, training = False):
        pass