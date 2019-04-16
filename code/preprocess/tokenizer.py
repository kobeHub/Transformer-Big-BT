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


from typing import List, Tuple, Iterator


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


# set of char used in the _escape_token() function
_ESCAPE_CHARS = set(u'\\_u;0123456789')
_UNESCAPE_REGEX = re.comlile(r'\\u|\\\\|\\([0-9]+);')

# min_count is the minimum of the token should appear before it can
# be added into vocab. The value is found via binary search to obtiain
# target vocabulary size
_MIN_MAX_COUNT = 1   # min value to use when bianry search for min_count
_MAX_MIN_COUNT = 100 # max value to use when binary search for min_count




######################### Define Subtokenizer class #################################

class Subtokenizer:
    """Encodes and decodes strings to/from integer IDs"""

    def __init__(self, vocab_file, reversed_tokens=RESERVED_TOKENS) -> Subtokenizer:
        """Create a vocab according to the given file"""
        tf.logging.info('Initializing Subtokenizer from file {}'.format(vocab_file))
        # todo 

    @staticmethods
    def vocab_from_files(vocab_file: str, files: List[str], target_vocab_size: int, 
            threshold: int, min_count=None, file_limit=1e6, 
            reserved_tokens=RESERVED_TOKENS) -> Subtokenizer:
        """Initializing subtoken vocabulary from files and save vocab in `vocab_file`
        
        Args:
            vocab_file: The file name to store the vocab
            files: List of files to generate vocab
            target_vocab_size: generate vocab size
            threshold: the threshold vocab can accept
            min_count: The minimum time of a subtoken should appear before added into 
                        vocabulary.
            file_limit: The maximum bytes of the sample text.
            reserved_tokens: RESERVED_TOKENS 
        """
        if tf.gfile.Exists(vocab_file):
            tf.logging.info('Vocab file {} already exists!!'.format(vocab_file))
        else:
            tf.logging.info('Generating vocab from files...')
            counts = _count_tokens(files, file_limit)
            alphabet = _generate_alphabet_dict(counts)
            subtoken_list = _generate_subtokens_with_target_vocab_size(
                    counts, alphabet, target_vocab_size, threshold, min_count,
                    reserved_tokens)
            tf.logging.info('Created vocab with {} subtokens.'.format(
                len(subtoken_list)))
            _save_vocab(vocab_file, subtoken_list)
        return Subtokenizer(vocab_file)

    def encode(self, raw_strin: str, add_eos=False) -> List[int]:
        """Encode string into list of int"""
        res = []
        # todo// tokens = _split_string_to_tokens()
        for token in tokens:
            res.extend(self._token_to_subtoken_ids(token))
        if add_eos:
            res.append(EOS_ID)
        return res

    def decode(self, subtokens: List[int]) -> str:
        """Decode from list of int"""
        if isinstance(np.adarray):
            subtokens = subtokens.toList()
        if not subtokens:
            return ''

        assert isinstance(subtokens, list) and isinstance(subtokens[0], int), (
        "Subtokens argument passed into decode() must be a list of integers.")

        return _join_tokens_to_string(self._subtoken_ids_to_tokens(subtokens))

    def _token_to_subtoken_ids(self, token: str) -> List[int]:
        """Encode a single token into a list of subtoken ids"""
        cache_loca = hash(token) % self._cache_size
        cache_k, cache_v = self._cache[cache_loca]
        if token == hash_k:
            return cache_v

        res = _split_string_to_tokens(
                _escape_token(token, self.alphabet), self.subtoken_to_id_dict,
                self.max_subtoken_length)
        res = [self.subtoken_to_ids_dict[subtoken_id] for subtoken_id in res]
        return res

    def _subtoken_ids_to_tokens(self, subtokens: List[int]) -> List[str]:
        """Convert a list of subtoken ids into a list of string tokens"""
        escaped_tokens = ''.join([
            self.subtoken_list[s] for s in subtokens
            if s < len(self.subtoken_list)])
        escaped_tokens = escaped_tokens.split('_')

        res = []
        for token in escaped_tokens:
            if token:
                res.append(_unescape_token(token))
        return res


############################# Functions for utils ##################################

def _save_vocab(vocab_file, subtoken_list) -> None:
    """Save subtokens into file"""
    with tf.gfile.Open(vocab_file, 'w') as f:
        for token in subtoken_list:
            f.write('{}\n'.format(token)



def _load_vocab_file(vocab_file, reserved_tokens=RESERVED_TOKENS) -> List[str]:
    """Load subtokens and ensures the reserved tokens are at top"""
    subtoken_list = []
    with tf.gfile.Open(vocab_file, mode='r') as f:
        for line in f:
            subtoken = line.strip()
            subtoken = subtoken[1:-1]
            if subtoken in reserved_tokens:
                continue
            subtoken_list.append(subtoken)

    return reserved_tokens + subtoken_list



def _


