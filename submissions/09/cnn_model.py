#!/usr/bin/env python3
import os
import re

from typing import Iterable
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

class CNNModel(tf.keras.Model):
    """ 
    Abstraction model class for creating Convolutional Neural Networks.
    """
  
    def __init__(self, input_shape: Iterable[int], out_shape: int, architecture_instructions: str, dim = 2, logdir: str = None):
        """Abstraction model class for creating Convolutional Neural Networks.

        Args:
            input_shape (Shape): Input shape.
            out_shape (Shape): Output shape.
            architecture_instructions (str): Instructions for creating the architecture of CNN model.
            dim (int, optional): Dimension of convolution layers.
            logdir (str, optional): Tensorboard logging directory if not specified Tesorboard is disabled. Defaults to None.
            --- 
            
            Specification of architecture_instructions:
            
            architecture_instructions argument, contains comma-separated specifications of the following layers:
                * C-(filters)-(kernelsize)-(stride)-(padding): Add a convolution layer with ReLU activation and 
                    specified number of filters, kernel size, stride and padding. 
                Example: C-10-3-1-same
                    
                * CB-(filters)-(kernelsize)-(stride)-(padding): Same as C-(filters)-(kernelsize)-(stride)-(padding), 
                    but use batch normalization. In detail, start with a convolution layer without bias and activation,
                    then add batch normalization layer, and finally ReLU activation.     
                Example: CB-10-3-1-same
                
                * M-(poolsize)-(stride): Add max pooling with specified size and stride, using the default "valid" padding. 
                Example: M-3-2
                
                * R-(layers): Add a residual connection. The layers contain a specification of at least one 
                    convolution layer (but not a recursive residual connection R). 
                    The input to the R layer should be processed sequentially by layers, 
                    and the produced output (after the ReLU non-linearity of the last layer) should
                    be added to the input (of this R layer). 
                Example: R-[C-16-3-1-same,C-16-3-1-same]
                
                * F: Flatten inputs. Must appear exactly once in the architecture.
                
                * H-(hiddenlayersize): Add a dense layer with ReLU activation and specified size. 
                Example: H-100
                
                * D-(dropoutrate): Apply dropout with the given dropout rate. 
                Example: D-0.5
                
            An example architecture might be --cnn=CB-16-5-2-same,M-3-2,F,H-100,D-0.5. You can assume the resulting network is valid; it is fine to crash if it is not.

        """
        inputs = tf.keras.layers.Input(shape=input_shape)

        next_inputs = inputs
        cnn_architecture = self.parse_architecture(architecture_instructions)
        
        if dim == 1:
          self.conv = tf.keras.layers.Convolution1D
          self.max_pool = tf.keras.layers.MaxPool1D
        elif dim == 2:
          self.conv = tf.keras.layers.Convolution2D
          self.max_pool = tf.keras.layers.MaxPool2D
        elif dim == 3:
          self.conv = tf.keras.layers.Convolution3D
          self.max_pool = tf.keras.layers.MaxPool3D
        else:
          raise ValueError('dim argument can have values 1, 2 or 3')

        for layer_type, parameters in cnn_architecture:
            next_inputs = self.apply_layer(next_inputs, layer_type, parameters)

        hidden = next_inputs

        if out_shape is not None:
            # Add the final output layer
            outputs = tf.keras.layers.Dense(out_shape, activation=tf.nn.softmax)(hidden)
        else:
            outputs = hidden

        super().__init__(inputs=inputs, outputs=outputs)
        
        if logdir is not None:
            self.tb_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        
            
    def apply_layer(self, inputs: list, layer_type: str, parameters: list):
        if layer_type == 'C':
            filters, kernel_size, stride, padding = parameters
            return self.conv(
                int(filters), int(kernel_size), int(stride), padding, activation=tf.nn.relu)(inputs)

        elif layer_type == 'CB':
            filters, kernel_size, stride, padding = parameters
            conv_outputs = self.conv(int(filters), int(kernel_size), int(stride), padding, use_bias=False)(inputs)
            batch_norm_outputs = tf.keras.layers.BatchNormalization()(conv_outputs)
            return tf.keras.activations.relu(batch_norm_outputs)

        elif layer_type == 'M':
            pool_size, stride = parameters
            return self.max_pool(int(pool_size), int(stride))(inputs)

        elif layer_type == 'R':
            res_block_instructions = parameters[0].replace('[','').replace(']', '')
            res_block_architecture = self.parse_architecture(res_block_instructions)
            next_inputs = inputs
            for res_layer_type, res_parameters in res_block_architecture:
                next_inputs = self.apply_layer(next_inputs, res_layer_type, res_parameters)
            return next_inputs + inputs

        elif layer_type == 'F':
            return tf.keras.layers.Flatten()(inputs)

        elif layer_type == 'H':
            [hidden_layer_size] = parameters
            return tf.keras.layers.Dense(int(hidden_layer_size), activation=tf.nn.relu)(inputs)

        elif layer_type == 'D':
            [dropout_rate] = parameters
            return tf.keras.layers.Dropout(float(dropout_rate))(inputs)
            
        else:
            raise ValueError(
                f'Unknown layer type "{layer_type}" found!')
    
    def parse_architecture(self, instructions_str: str):
        instructions = []
        
        instructions_parts = re.split(r",(?!(?:[^,\[\]]+,)*[^,\[\]]+])", instructions_str, 0)

        for ins_part in instructions_parts:
            [layer_type, *parameters] = re.split(r"-(?!(?:[^-\[\]]+-)*[^-\[\]]+])", ins_part, 0)
            instructions.append((layer_type, parameters))

        return instructions
    