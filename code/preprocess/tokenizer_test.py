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





if __name__ == '__main__':
    unittest.main()
