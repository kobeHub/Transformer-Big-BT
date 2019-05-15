"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Tue 30 Apr 2019

 Transformer model utilities methods. 
"""

import tensorflow as tf
import numpy as np

_NEG_INF = -1e9



def position_encoding(length, hidden_size, min_timescale=1.0, 
        max_timescale=1.0e4):
    """Encodeing the position.
    
    The position encoding is computed as the mix of sine and cosine 
    funtions.

    Args:
        length: sequence length
        min_timescale: minimum scale appiled at each position
        max_timescale: maximum scale applied at each position

    @r:
        Tensor with shape [length, hidden_size]
    """
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = hidden_size // 2
    log_timescales_increment = (
            np.log(float(max_timescale) / float(min_timescale)) /
            (tf.cast(num_timescales, tf.float32) - 1))

    inv_timescales = min_timescale * tf.exp(tf.cast(tf.range(num_timescales), 
        tf.float32) * -log_timescales_increment )
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

    return signal


def get_padding(x, padding_value=0):
    """Get float tensor representing the padding of x. which 
    0 -> nonpading, 1 -> padding.
    """
    return tf.cast(tf.equal(x, padding_value), tf.float32)



def get_padding_bias(x):
    """Calcuate bias tensor from padding values in tensor.
    
    Args:
        x: int tensor shape [batch_size, length]

    @r:
        shape [batch_size, 1, 1, length]
    """
    with tf.name_scope('attention_bias'):
        padding = get_padding(x)
        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(
                tf.expand_dims(attention_bias, axis=1), axis=1)
        return attention_bias



def get_decoder_self_attention_bias(length):
    """Calculate bias for decoder that maintains model's autoregression property.
    
    Args:
        length: int length of seq
    @r:
        float tensor [1, 1, length, length]
    """
    with tf.name_scope('decoder_self_attention_bias'):
        valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = _NEG_INF * (1.0 - valid_locs) 
    return decoder_bias
