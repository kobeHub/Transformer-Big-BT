"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Mon Apr 29 CST 2019

 Implment the embedding layers with shared weights
"""

import tensorflow as tf
from code.model import model_utils




class EmbeddingSharedWeights(tf.layers.Layer):
    """Calculate input embeddings and pre-softmax linear with shared weights"""

    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self):
        with tf.variable_scope('embedding_and_softmax', reuse=tf.AUTO_REUSE):
            # The initialized weights
            self.shared_weights = tf.get_variable('weights',
                    [self.vocab_size, self.hidden_size],
                    initializer=tf.random_normal_initializer(
                        0., self.hidden_size ** -0.5))
        self.built = True

    def call(self, x):
        """Get token embeddings of x
        
        Args: 
            x: input int64 tensor shape [batch_size, length]
        @r:
            embeddings: float32 tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope('embeddings'):
            # Mask shape [batch_size, length]
            mask = tf.cast(tf.not_equal(x, 0), tf.float32)

            embeddings = tf.gather(self.shared_weights, x)
            embeddings *= rd.expand_dims(mask, -1)
            embeddings *= self.hidden_size ** 0.5

            return embeddings

    def linear(self, x):
        """Get the logits of the embeddings layer.
        
        Args:
            x: input float32 tensor shape [batch_size, length, hidden_size]
        @r: 
            float32 tensor shape [batch_size, length, vocab_size]
        """
        with tf.name_scope('presoftmax_linear'):
            batch_size =  tf.shape(x)[0]
            length = tf.shape(x)[1]

            x = tf.reshape(x, [-1, self.hidden_size])
            logits = tf.matmul(x, self.shared_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])
        
        
