import tensorflow as tf


class Head(tf.keras.Model):
    def __init__(self, input_size, output_size):
        inputs = tf.keras.Input(shape=[None, None, input_size])
        
        x = inputs 
        
        kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
        for _ in range(4):
            x = tf.keras.layers.Conv2D(input_size, 3, padding="same", kernel_initializer=kernel_init)(x)
            x = tf.keras.layers.ReLU()(x)
        outputs = tf.keras.layers.Conv2D(output_size,3,1,padding='same')(x)
      
        super().__init__(name="ClassificationSubNet", inputs=inputs, outputs=outputs)
      