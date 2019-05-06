"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Mon Apr 22 2019
 
 Implement Feed-Forward Network 
"""

import tensorflow as tf


class FeedForwardNetwork(tf.layers.Layer):
    """fully-connected feedforward network"""

    def __init__(self, hidden_size: int, filter_size: int, relu_dropout: float, 
            trainable: bool, allow_pad: bool):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.trainable = trainable
        self.allow_pad = allow_pad

        self.filter_dense_layer = tf.layers.Dense(filter_size, 
                use_bias=True, activation=tf.nn.relu, name='filter_dense')
        self.output_dense_layer = tf.layers.Dense(hidden_size,
                use_bias=True, name='output_layer')


    def call(self, x, padding=None):
        """Compuate the output of ffn.
        
        x: input tensor shape: [batch_size, length, hidden_size]
        padding: the content padding in x, if not None, it will be remove 
                from x while compuating. And add back after computation

        @r: [batch_size, length, hidden_size]
        """
        padding = None if not self.allow_pad else padding

        batch_size = tf.shape(x)[0]      # Get shape tensor  //   use x.shape get Demension object
        length = tf.shape(x)[1]

        if padding is not None:
            with tf.name_scope('remove_padding'):
                pad = tf.reshape(padding, [-1])
                nonpad_ids = tf.cast(tf.where(pad < 1e-9), tf.int32)

                # Reshape x to [batch_size*length, hidden_size] to remove padding
                x = tf.reshape(x, [-1, self.hidden_size])
                x = tf.gather_nd(x, indices=nonpad_ids)

                # Reshape x to 3 dimension
                x.set_shape([None, self.hidden_size])
                x = tf.expand_dims(x, axis=0)

        output = self.filter_dense_layer(x)
        if self.trainable:
            output = tf.nn.dropout(output, rate=self.relu_dropout)
        output = self.output_dense_layer(output)

        if padding is not None:
            with tf.name_scope("re_add_padding)"):
                output = tf.squeeze(output, axis=0)
                output = tf.scatter_nd(indices=nonpad_ids,
                        updates=output,
                        shape=[batch_size*length, self.hidden_size])
                output = tf.reshape(output, [batch_size, length, self.hidden_size])

        return output
                
    

        
