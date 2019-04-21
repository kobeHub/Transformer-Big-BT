"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Sun Apr 21 20:43:37 CST 2019

 Unittest for the tokenizer utils function
"""
import unittest

from tokenizer import _split_string_to_tokens

class TestTokenizerUtils(unittest.TestCase):

    def test_split_string_to_tokens(self):
        self.assertEqual(_split_string_to_tokens("just A T cse's use"), ['just', 'A', 'T', "cse's", 'use'])
        self.assertEqual(_split_string_to_tokens('还可以啊', 'zh'), ['还', '可', '以', '啊'])
        self.assertEqual(_split_string_to_tokens('欢迎Alice Let\'s go', 'zh'), 
                ['欢','迎','Alice','Let\'s', 'go'])
        self.assertEqual(_split_string_to_tokens('Michale, 你为什么'), 
                ['Michale', '你', '为','什', '么'])


if __name__ == '__main__':
    unittest.main()
