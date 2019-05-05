"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Sun Apr 21 20:43:37 CST 2019

 Unittest for the tokenizer utils function
"""
import unittest
import tensorflow as tf 
tf.logging.set_verbosity(tf.logging.INFO)



from tokenizer import _split_string_to_tokens
from tokenizer import _join_tokens_to_string
from tokenizer import Tokenizer
from tokenizer import _escape_token, _unescape_token
from tokenizer import _ALPHANUMERIC_CHAR_SET


import os 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data/UMcorpus/processed')
VOCAB_FILE = os.path.join(DATA_DIR, 'vocab.ende.610390')

class TestTokenizerUtils(unittest.TestCase):

    def test_split_string_to_tokens(self):
        self.assertEqual(_split_string_to_tokens("just A T cse's use"), ['just', 'A', 'T', "cse's", 'use'])
        self.assertEqual(_split_string_to_tokens('还可以啊', 'zh'), ['还', '可', '以', '啊'])
        self.assertEqual(_split_string_to_tokens('欢迎Alice Let\'s go, 还不错吧', 'zh'), 
                ['欢','迎','Alice','Let\'s', 'go', ', ', '还', '不', '错', '吧'])
        self.assertEqual(_split_string_to_tokens('Michale, 你为什么'), 
                ['Michale', ', ', '你', '为','什', '么'])
        print(_split_string_to_tokens('(a) Just a test'))


    def test_join_token_to_string(self):
        self.assertEqual(_join_tokens_to_string(['Just', 'a', 'test']), 'Just a test')
        self.assertEqual(_join_tokens_to_string(['只', '是', '以', 'Alice'], 'zh'), '只是以Alice')
        self.assertEqual(_join_tokens_to_string(['All', 'right', '?']), 'All right?')

    def test_Subtokenizer(self):
        tokenizer = Tokenizer.vocab_from_files('vocab.ende.', ['./vocab.test'])
        print(tokenizer.token_list)
        print(tokenizer.token_to_id_dict)
        print(tokenizer.vocab_size)
        self.assertEqual(len(tokenizer.token_list), tokenizer.vocab_size)

    def test_encode_decode(self):
        tokenizer = Tokenizer('vocab.ende.46')
        str1 = 'what you seen locked into?'
        str2 = '路都是行者，what?'
        res1 = tokenizer.encode(str1, add_eos=True)
        str11 = tokenizer.decode(res1, add_eos=True)
        res2 = tokenizer.encode(str2, add_eos=True)
        str21 = tokenizer.decode(res2, add_eos=True, type_='zh')
        print(str1, str2)
        print(res1, res2)
        print(str11, str21)
        self.assertEqual(str1, str11)
        self.assertEqual(str2, str21)

    def test_escape_token(self):
        str1 = '\uffef('
        str2 = '\u1261'
        str3 = '我'
        res1 = _escape_token(str1, _ALPHANUMERIC_CHAR_SET)
        res2 = _escape_token(str2, _ALPHANUMERIC_CHAR_SET)
        res3 = _escape_token(str3, _ALPHANUMERIC_CHAR_SET)
        print()
        print('Escape char test:')
        print(res1)
        print(res2)
        print(res3)

        print('Unescape char test:')
        str11 = _unescape_token(res1)
        str21 = _unescape_token(res2)
        str31 = _unescape_token(res3)
        print(str1, str11)
        print(str2, str21)
        print(str3, str31)


    def test_dataset_decode(self):
        tokenizer = Tokenizer(VOCAB_FILE)
        source_ids = [254939, 495884, 494380,  59409, 177392, 580889, 280523, 224280,
       388662, 460449,  24656, 493847, 572291, 407836, 554070, 285057,
       596466,   8485, 183316, 275391, 421102,  34097, 410489, 601320,
            1,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0]
        target_ids = [222842, 595004, 608916,  41411, 437579, 484813, 597251,  48875,
       437579, 425240, 305538, 116475, 597251, 567015,  32418, 484813,
       301587,  60117, 178244, 552052, 462251, 284775, 587092, 543810,
       608916,  27637, 191835, 456038, 317479, 209102,  19162, 559860,
       467261, 467056, 587092, 170418, 348904, 102539, 329258, 328805,
       216408, 399803, 484813, 587092, 317479,  40909, 371496, 173398,
       384795, 449458, 597251, 449458, 413263,   5051,   5334,  21244,
       266473,  27637,  27665, 163260, 368259, 587172,      1,      0,
            0,      0]
        s1 = tokenizer.decode(source_ids, add_eos=True)
        s2 = tokenizer.decode(target_ids, add_eos=True, type_='zh')
        print(s1)
        print(s2)


if __name__ == '__main__':
    unittest.main()
