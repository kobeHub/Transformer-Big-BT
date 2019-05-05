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
from code.train_and_evaluate import run_transformaer


# Define default dir args
umcorpus_raw = os.path.join(BASE_DIR, 'data/UMcorpus/RAW')
eval_dir = os.path.join(umcorpus_raw, 'data/Testing')
umcorpus_data = os.path.join(BASE_DIR, 'data/UMcorpus/processed')

def usage_test(name: str='') -> None:
    basic_usage(name)

def pre_data(raw_dir=umcorpus_raw, eval_dir=eval_dir, data_dir=umcorpus_data, shuffle=True) -> None:
    print('Run data preprocess...')
    process(raw_dir, eval_dir, data_dir, shuffle)

def train():
    print('Begin to train and eval Transformer model...')




if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    fire.Fire()
