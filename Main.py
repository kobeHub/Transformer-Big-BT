"""
Author: Inno Jia @ https://kobehub.github.io

The basic command line tool for the NMT project.
"""

import fire
import sys
sys.path.append('..')

 
from code.preprocess.tokenizer import split_zh_string_to_tokens

def usage_test(name: str='') -> None:
    basic_usage(name)

def pre_data() -> None:
    print('Run data process...')

def train() -> None:
    print('Begin to train the model...')

def tokiner_test(string: str) -> None:
    t = split_zh_string_to_tokens(string)
    print(t)


if __name__ == '__main__':
    fire.Fire()
