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
from code.translate import translate_main
from code.train_and_evaluate import run_transformaer


# Define default dir args
umcorpus_raw = os.path.join(BASE_DIR, 'data/UMcorpus/RAW')
eval_dir = os.path.join(umcorpus_raw, 'data/Testing')
umcorpus_data = os.path.join(BASE_DIR, 'data/UMcorpus/processed')
model_dir = os.path.join(BASE_DIR, 'saved_model')
export_dir = os.path.join(BASE_DIR, 'exported')

vocab_file = os.path.join(umcorpus_data, 'vocab.ende.610390')
bleu_source = os.path.join(umcorpus_data, 'bleu_source.txt')
bleu_ref = os.path.join(umcorpus_data, 'bleu_ref.txt')


# Define hooks 
HOOKS = ['loggingtensorhook', 'loggingmetrichook', 'profilerhook']


# args for translate cli 
args_translate_cli = {
        'params_set': 'base',
        'vocab_file': vocab_file,
        'model_dir': model_dir}



def usage_test(name: str='') -> None:
    basic_usage(name)


def pre_data(raw_dir=umcorpus_raw, eval_dir=eval_dir, data_dir=umcorpus_data, shuffle=True) -> None:
    print('Run data preprocess...')
    process(raw_dir, eval_dir, data_dir, shuffle)


def train(bleu_source, bleu_ref, num_gpus=1, params_set='base', 
        data_dir=umcorpus_data, model_dir=model_dir, 
        export_dir=export_dir, batch_size=None, allow_ffn_pad=True, 
        hooks=HOOKS, stop_threshold=15.0, vocab_file=vocab_file):
    print('Begin to train and eval Transformer model...')
    
    run_transformaer(num_gpus=num_gpus, params_set=params_set, data_dir=data_dir, 
            model_dir=model_dir, export_dir=export_dir, batch_size=batch_size, 
            allow_ffn_pad=allow_ffn_pad, bleu_source=bleu_source, bleu_ref=bleu_ref, 
        hooks=hooks, stop_threshold=stop_threshold, vocab_file=vocab_file)


def translate(text=None, inputs_file=None, output_file=None, args=args_translate_cli):
    translate_main(text, inputs_file, output_file, args)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    fire.Fire()
