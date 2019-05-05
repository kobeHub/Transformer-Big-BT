"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Mon Apr 15 2019

 Defines Subtokenizer class to encode and decode strings
"""

import collections
import re
import sys

import tensorflow as tf
import numpy as np
import six
import unicodedata


import sys
import os
sys.path.append('..')
from typing import List, Tuple, Iterator, Dict
from compute_bleu import uregex


#################### super constant ###########################
# Define the reserved tokens
PAD = '<PAD>'
PAD_ID = 0
EOS = '<EOS>'
EOS_ID = 1
RESERVED_TOKENS = [PAD, EOS]

# All characters contains all letter ans number
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))

_IS_CHINESE_CHAR = lambda c: c >= u'\u4e00' and c <= u'\u9fa5'


# set of char used in the _escape_token() function
_ESCAPE_CHARS = set(u'\\_u;0123456789')
_UNESCAPE_REGEX = re.compile(r'\\u|\\\\|\\([0-9]+);')
_UNDEFINED_UNICODE = u'\u3013'

# Punctuations and symbols shall be in the vocab
_PUNCTUATIONS = uregex.property_chars('P')
_SYMBOLS = uregex.property_chars('S')


######################### Define Subtokenizer class #################################

class Tokenizer:
    """Encodes and decodes strings to/from integer IDs"""

    def __init__(self, vocab_file, reserved_tokens=RESERVED_TOKENS):
        """Create a vocab according to the given file"""
        tf.logging.info('Initializing Tokenizer from file {}'.format(vocab_file))

        # In order to use Chinese, token based on word level
        self.token_list = _load_vocab_file(vocab_file, reserved_tokens)
        self.token_to_id_dict = _list_to_index_dict(self.token_list)
        self.vocab_size = len(self.token_to_id_dict)

        # todo: create cache to speed tokenization


    @staticmethod
    def vocab_from_files(vocab_file: str, files: List[str],
            reserved_tokens=RESERVED_TOKENS):
        """Initializing subtoken vocabulary from files and save vocab in `vocab_file`

        Args:
            vocab_file: The file name to store the vocab
            files: List of files to generate vocab
            reserved_tokens: RESERVED_TOKENS
        """
        if tf.gfile.Exists(vocab_file):
            tf.logging.info('Vocab file {} already exists!!'.format(vocab_file))
        else:
            tf.logging.info('Generating vocab from files...')
            counts = _count_tokens(files)
            token_list = list(counts.keys())
            tf.logging.info('Created vocab with {} tokens.'.format(
                len(token_list)))
            vocab_file += str(len(token_list))
            _save_vocab(vocab_file, token_list)
        return Tokenizer(vocab_file)

    def encode(self, raw_string: str, add_eos=False) -> List[int]:
        """Encode string into a list of int"""
        res = []
        tokens = _split_string_to_tokens(raw_string)

        
        for token in tokens:
            try:
                token = token.strip()
                res.append(self.token_to_id_dict[token])
            except KeyError:
                break 
        if add_eos:
            res.append(EOS_ID)
        return res

    def decode(self, token_ids: List[int], add_eos=False, type_=None) -> str:
        """Decode from list of int
        type_ = 'zh' to decode Chinese
        """
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.toList()
        if not token_ids:
            return ''

        if add_eos:
            token_ids = token_ids[:-1]

        assert isinstance(token_ids, list) and isinstance(token_ids[0], int), (
        "Subtokens argument passed into decode() must be a list of integers.")

        return _join_tokens_to_string(self._token_ids_to_tokens(token_ids), type_=type_)

    def _token_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert a list of subtoken ids into a list of string tokens"""
        res = [self.token_list[i] for i in token_ids if i < self.vocab_size]

        return res

    


############################# Functions for utils ##################################

def _split_string_to_tokens(text: str, type_=None) -> List[str]:
    """Splites English text to a list of str tokens"""
    if not text:
        return []

#    if type_ == 'zh':
#        res = []
#        is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
#        word = ''
#        for i, token in enumerate(text):
#            if is_alnum[i]:
#                if _IS_CHINESE_CHAR(token):
#                    if word:
#                        res.append(word)
#                        word = ''
#                    res.append(token)
#                else:
#                    word += token
#
#        return res

    res = []
    start = 0
    is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
    #if is_alnum[0] and _IS_CHINESE_CHAR(text[0]):
    #    res.append(text[0])

    for pos in range(1, len(text)):
        if is_alnum[pos-1] and _IS_CHINESE_CHAR(text[pos-1]):
            res.append(text[pos-1])
            start = pos
            continue

        if is_alnum[pos] != is_alnum[pos-1] and text[pos] != "'" and text[pos-1] != "'":
            token = text[start:pos]
            if token != u' ' or start == 0:
                res.append(token)
            start = pos

    res.append(text[start:])
    return res



def _save_vocab(vocab_file, subtoken_list):
    """Save subtokens into file"""
    with tf.gfile.Open(vocab_file, 'w') as f:
        for token in subtoken_list:
            f.write('{}\n'.format(token.strip()))



def _load_vocab_file(vocab_file, reserved_tokens=RESERVED_TOKENS) -> List[str]:
    """Load subtokens and ensures the reserved tokens are at top"""
    token_list = []
    with tf.gfile.Open(vocab_file, mode='r') as f:
        for line in f:
            token = line.strip()
            #token = token[1:-1]
            if token in reserved_tokens:
                continue
            token_list.append(token)

    return reserved_tokens + token_list



def _join_tokens_to_string(tokens, type_: str=None) -> str:
    """Join a list of string tokens into a single string.
        type: zh or en
    """
    if type_ == 'zh':
        return ''.join(tokens)
    else:
        token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
        res = []
        for i, token in enumerate(tokens):
            if i > 0 and token_is_alnum[i-1] and token_is_alnum[i]:
                res.append(u' ')
            res.append(token)
        return ''.join(res)



def _count_tokens(files: List[str]) -> Dict[str, int]:
    """Count token in files.

    Samples file_limit bytes from each file, and counts the words that appear
    in the samples. The samples are semi-evenly distributed across the file.
    """
    token_counts = collections.defaultdict(int)

    for file_path in files:
        with tf.gfile.Open(file_path, mode='r') as f:
            for line in f.readlines():
                for token in _split_string_to_tokens(line):
                    token_counts[token] += 1

    return token_counts


def _list_to_index_dict(lst):
    return {item: n for n, item in enumerate(lst)}



def _escape_token(token: str, alphabet: List[str]) -> str:
    r"""Replace characters which aren't in the alphabet and append '_' to token

    Apply three transformations to the token:
        1. Replace underline character "_" with "\u", and backslash "\" with "\\".
        2. Replace characters outside of the alphabet with "\###;", where ### is the
            character's Unicode code point.
        3. Appends "_" to mark the end of a token.

    """
    token = token.replace(u'\\', u'\\\\').replace(u'_', u'\\u')
    ret = [c if c in alphabet and c != u"\n" else r"\%d;" % ord(c) for c in token]
    return u"".join(ret) + "_"


def _unescape_token(token: str) -> str:
    r"""Replace escaped characters in the token with their unescaped version

        Applies inverse transformations as _escape_token():
            1. Replace "\u" with "_", and "\\" with "\".
            2. Replace "\###;" with the unicode character the ### refers to.
    """
    def match(m):
        if m.group(1) is None:
            return u'_' if m.group(0) == u'\\u' else u'\\'

        try:
            return six.unichr(int(m.group(1)))
        except (ValueError, OverflowError) as _:
            return _UNDEFINED_UNICODE

    # Use match function to replace escaped substrings in the token.
    return _UNESCAPE_REGEX.sub(match, token)

