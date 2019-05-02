"""
 Date: Thu 02 May 2019
 
 Compute BLEU score.
"""

import re
import sys
import unicodedata

import tensorflow as tf
import metrics


class UnicodeRegex(object):
    """Recognize all punctuation and symbols"""

    def __init__(self):
        punctuation = self.property_chars('P')
        self.nondigit_punct_re = re.compile(r'([^\d])([' + punctuation + r'])')
        self.punct_nondigit_re = re.compile(r'([' + punctuation + r'])([^\d])')
        self.symbol_re = re.compile(r'([' + self.property_chars('S') + r'])')

    def property_chars(self, prefix):
        return ''.join(chr(x) for x in range(sys.maxunicode)
                if unicodedata.category(chr(x)).startswith(prefix))



uregex = UnicodeRegex()



def bleu_tokenize(string):
    """The input strings are expected as a single line.
    Just tikenize on punctuation and symbols, except when a punctuation
    is proceded and followed by a digit. 
    """
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    return string.split()



def bleu_wrapper(ref_file, hyp_file, case_sensitive=False):
    ref_lines = tf.gfile.Open(ref_file).read().strip().splitlines()
    hyp_lines = tf.gfile.Open(hyp_file).read().strip().splitlines()

    if len(ref_lines) != len(hyp_lines):
        raise ValueError("Reference and translation files not same size")

    if not case_sensitive:
        ref_lines = [x.lower() for x in ref_lines]
        hyp_lines = [x.lower() for x in hyp_lines]
    ref_tokens = [bleu_tokenize(x) for x in ref_lines]
    hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]

    return metrics.compute_bleu(ref_tokens, hyp_tokens) * 100



def bleu_results(ref_file: str, hyp_file: str, case_sensitive: bool):
    tf.logging.info('Computing the BLEU...')
    score = bleu_wrapper(ref_file, hyp_lines, case_sensitive)
    if case_sensitive:
        tf.logging.info('Case sensitive results: {}'.format(score))
    else:
        tf.logging.info('Case insensitive results: {}'.format(score))


