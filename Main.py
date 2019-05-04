"""
Author: Inno Jia @ https://kobehub.github.io

The basic command line tool for the NMT project.
"""

import fire
import sys
sys.path.append('..')


import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import tensorflow as tf

from code.preprocess.pre_data import process



# Define default dir args
umcorpus_raw = os.path.join(BASE_DIR, 'data/UMcorpus/RAW')
eval_dir = os.path.join(umcorpus_raw, 'Testing')
umcorpus_data = os.path.join(BASE_DIR, 'data/UMcorpus/processed')

def usage_test(name: str='') -> None:
    basic_usage(name)

def pre_data(raw_dir=umcorpus_raw, eval_dir=eval_dir, data_dir=umcorpus_data) -> None:
    print('Run data preprocess...')
    process(raw_dir, eval_dir, data_dir)

def train() -> None:
    print('Begin to train the model...')



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    fire.Fire()
