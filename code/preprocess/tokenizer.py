"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Mon Apr 15 2019
 
 Defines Subtokenizer class to encode and decode strings
"""

import clooections
import re
import sys

import tensorflow as tf
import numpy as np


PAD = '<PAD>'
PAD_ID = 0
EOS = '<EOS>'
RESERVED_TOKENS = [PAD, EOS]

# All characters contains all letter ans number
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in xrange(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))


class Subtokenizer:
    """Encodes and decodes strings to/from integer IDs"""

    def __init__(self, vocab_file, reversed_tokens=None):
        """Create a vocab according to the given file"""
        tf.logging.info('Initializing Subtokenizer from file {}'.format(vocab_file))

        
