"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Wed 01 May 2019 09:05:58 PM CST

 Test case for beam search
"""

import tensorflow as tf
import beam_search

class BeamSearchHelperTests(tf.test.TestCase):

  def test_expand_to_beam_size(self):
    x = tf.ones([7, 4, 2, 5])
    x = beam_search._expand_to_beam_size(x, 3)
    with self.test_session() as sess:
      shape = sess.run(tf.shape(x))
    self.assertAllEqual([7, 3, 4, 2, 5], shape)

  def test_shape_list(self):
    y = tf.placeholder(dtype=tf.int32, shape=[])
    x = tf.ones([7, y, 2, 5])
    shape = beam_search._shape_list(x)
    self.assertIsInstance(shape[0], int)
    self.assertIsInstance(shape[1], tf.Tensor)
    self.assertIsInstance(shape[2], int)
    self.assertIsInstance(shape[3], int)

  def test_get_shape_keep_last_dim(self):
    y = tf.constant(4.0)
    x = tf.ones([7, tf.cast(tf.sqrt(y), tf.int32), 2, 5])
    shape = beam_search._get_shape_keep_last_dim(x)
    self.assertAllEqual([None, None, None, 5],
                        shape.as_list())

  def test_flatten_beam_dim(self):
    x = tf.ones([7, 4, 2, 5])
    x = beam_search._flatten_beam_dim(x)
    with self.test_session() as sess:
      shape = sess.run(tf.shape(x))
    self.assertAllEqual([28, 2, 5], shape)

  def test_unflatten_beam_dim(self):
    x = tf.ones([28, 2, 5])
    x = beam_search._unflatten_beam_dim(x, 7, 4)
    with self.test_session() as sess:
      shape = sess.run(tf.shape(x))
    self.assertAllEqual([7, 4, 2, 5], shape)

  def test_gather_beams(self):
    x = tf.reshape(tf.range(24), [2, 3, 4])
    # x looks like:  [[[ 0  1  2  3]
    #                  [ 4  5  6  7]
    #                  [ 8  9 10 11]]
    #
    #                 [[12 13 14 15]
    #                  [16 17 18 19]
    #                  [20 21 22 23]]]

    y = beam_search._gather_beams(x, [[1, 2], [0, 2]], 2, 2)
    with self.test_session() as sess:
      y = sess.run(y)

    self.assertAllEqual([[[4, 5, 6, 7],
                          [8, 9, 10, 11]],
                         [[12, 13, 14, 15],
                          [20, 21, 22, 23]]],
                        y)

  def test_gather_topk_beams(self):
    x = tf.reshape(tf.range(24), [2, 3, 4])
    x_scores = [[0, 1, 1], [1, 0, 1]]

    y = beam_search._gather_topk_beams(x, x_scores, 2, 2)
    with self.test_session() as sess:
      y = sess.run(y)

    self.assertAllEqual([[[4, 5, 6, 7],
                          [8, 9, 10, 11]],
                         [[12, 13, 14, 15],
                          [20, 21, 22, 23]]],
                        y)


if __name__ == "__main__":
  tf.test.main()


