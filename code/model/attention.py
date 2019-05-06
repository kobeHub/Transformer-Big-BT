"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Mon Apr 22 2019

 Implementation of the multihead attention and self-attention layers
"""

import tensorflow as tf



class Attention(tf.layers.Layer):
    """Multi-Headed attention layers"""

    def __init__(self, hidden_size: int, num_heads: int, attention_dropout: float, train: bool):
        if hidden_size % num_heads != 0:
            raise ValueError('Hidden size must be divisible by'
                    ' num_heads')

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.train = train

        # layers for linearly projecting the queries, keys, values
        self.q_layer = tf.layers.Dense(hidden_size, use_bias=False, name='queries')
        self.k_layer = tf.layers.Dense(hidden_size, use_bias=False, name='keys')
        self.v_layer = tf.layers.Dense(hidden_size, use_bias=False, name='values')

        self.output_layers = tf.layers.Dense(hidden_size, use_bias=False, name='output_transform')


    def split_heads(self, x):
        """Split x into multi heads, and transpose the result.

        Args:
            x: Tensor with shape [batch_size, length, hidden_size]
        
        @r: A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope('split_heads'):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            depth = self.hidden_size // self.num_heads
            
            # split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been splited
        
            Args: 
                x: Tensor with shpae [batch_size, num_heads, length, hidden_size/num_heads]
            @r: A tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope('combine_heads'):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y, bias, cache=None):
        """Apply attention mechanism to x and y
        
        Args:
            x: a tensor with shape [batch_size, length_x, hidden_size]
            y: a tensor with shape [batch_size, length_y, hidden_size]
            bias: attention bias that will be added to the result of the dot product.
            cache: (Used during prediction) dictionary with tensors containing results
                    of previous attentions. The dictionary must have the items:
                    {"k": tensor with shape [batch_size, i, key_channels],
                    "v": tensor with shape [batch_size, i, value_channels]}
                    where i is the current decoded length.
        Returns:
            Attention layer output with shape [batch_size, length_x, hidden_size]
        """
        # Prepare linearly project to q, k, v
        q = self.q_layer(x)
        k = self.k_layer(y)
        v = self.k_layer(y)

        if cache:
            k = tf.concat([cache['k'], k], axis=1)
            v = tf.concat([cache['v'], v], axis=1)

            # update cache
            cache['k'] = k
            cache['v'] = v

        # split into heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # prevent dot product between q, k too large
        q *= (self.hidden_size // self.num_heads) ** -0.5

        # dot product
        logits = tf.matmul(q, k, transpose_b=True)
        logits += bias
        weights = tf.nn.softmax(logits, name='attention_weights')
        if self.train:
            weights = tf.nn.dropout(weights, rate=self.attention_dropout)
        attention_output = tf.matmul(weights, v)


        # combine heads
        attention_output = self.combine_heads(attention_output)
        attention_output = self.output_dense_layer(attention_output)

        return attention_output



class SelfAttention(Attention):
    """Multi-Heads attention layer"""

    def call(self, x, bias, cache=None):
        return super(SelfAttention, self).call(x, x, bias, cache)

