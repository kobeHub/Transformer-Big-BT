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



