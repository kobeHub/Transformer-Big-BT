"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Thu 02 May 2019

 Test case for compute_bleu
"""

import tempfile

import tensorflow as tf 

import compute_bleu


class ComputeBleuTest(tf.test.TestCase):

  def _create_temp_file(self, text):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with tf.gfile.Open(temp_file.name, 'w') as w:
      w.write(text)
    return temp_file.name

  def test_bleu_same(self):
    ref = self._create_temp_file("test 1 two 3\nmore tests!")
    hyp = self._create_temp_file("test 1 two 3\nmore tests!")

    uncased_score = compute_bleu.bleu_wrapper(ref, hyp, False)
    cased_score = compute_bleu.bleu_wrapper(ref, hyp, True)
    self.assertEqual(100, uncased_score)
    self.assertEqual(100, cased_score)

  def test_bleu_same_different_case(self):
    ref = self._create_temp_file("Test 1 two 3\nmore tests!")
    hyp = self._create_temp_file("test 1 two 3\nMore tests!")
    uncased_score = compute_bleu.bleu_wrapper(ref, hyp, False)
    cased_score = compute_bleu.bleu_wrapper(ref, hyp, True)
    self.assertEqual(100, uncased_score)
    self.assertLess(cased_score, 100)

  def test_bleu_different(self):
    ref = self._create_temp_file("Testing\nmore tests!")
    hyp = self._create_temp_file("Dog\nCat")
    uncased_score = compute_bleu.bleu_wrapper(ref, hyp, False)
    cased_score = compute_bleu.bleu_wrapper(ref, hyp, True)
    self.assertLess(uncased_score, 100)
    self.assertLess(cased_score, 100)

  def test_bleu_tokenize(self):
    s = "Test0, 1 two, 3"
    print(s)
    tokenized = compute_bleu.bleu_tokenize(s)
    print(tokenized)
    self.assertEqual(["Test0", ",", "1", "two", ",", "3"], tokenized)


if __name__ == "__main__":
  tf.test.main()

