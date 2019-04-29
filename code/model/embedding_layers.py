"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Mon Apr 29 CST 2019

 Implment the embedding layers with shared weights
"""

import tensorflow as tf
import model_utils



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
        with tf.name_scope()
        
