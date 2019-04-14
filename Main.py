"""
Author: Inno Jia @ https://kobehub.github.io

The basic command line tool for the NMT project.
"""

import fire
import sys
sys.path.append('..')

from code.preprocess.pre_data import basic_usage 


def usage_test(name: str='') -> None:
    basic_usage(name)

def pre_data() -> None:
    print('Run data process...')

def train() -> None:
    print('Begin to train the model...')



if __name__ == '__main__':
    fire.Fire()
