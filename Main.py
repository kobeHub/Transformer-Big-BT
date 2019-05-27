"""
Author: Inno Jia @ https://kobehub.github.io

The basic command line tool for the NMT project.
"""

import fire
import sys
sys.path.append('..')


import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/data/gradstu/Transformer'

import tensorflow as tf

from code.preprocess.pre_data import process
from code.train_and_evaluate import run_transformaer
from code.translate import translate_main
from code.train_and_evaluate import run_transformaer
from code.params_num import get_params_num

# Define default dir args
umcorpus_raw = os.path.join(BASE_DIR, 'data/UMcorpus/RAW')
eval_dir = os.path.join(umcorpus_raw, 'data/Testing')
umcorpus_data = os.path.join(BASE_DIR, 'data/UMcorpus/processed')
multiun_raw = os.path.join(BASE_DIR, 'data/MultiUN/RAW')
processed_data = os.path.join(BASE_DIR, 'data/processed')
opensubtitles_raw = os.path.join(BASE_DIR, 'data/OpenSubtitles/RAW')
wmtcasia15_raw = os.path.join(BASE_DIR, 'data/WMT18/casia2015')
wmtcasict15_raw = os.path.join(BASE_DIR, 'data/WMT18/casict2015')


model_dir_v0 = os.path.join(DATA_DIR, 'saved_model')
model_dir_v1 = os.path.join(DATA_DIR, 'saved_model_v1')
export_dir = os.path.join(DATA_DIR, 'exported')
graphs_dir = os.path.join(DATA_DIR, 'graphs')


vocab_file_v0 = os.path.join(umcorpus_data, 'vocab.ende.610390')
vocab_file_v1 = os.path.join(processed_data, 'vocab.ende.1132998')   # todo
bleu_source = os.path.join(processed_data, 'bleu_source.txt')
bleu_ref = os.path.join(processed_data, 'bleu_ref.txt')


# Define hooks 
HOOKS = ['loggingtensorhook', 'loggingmetrichook', 'profilerhook']


# args for translate cli 
args_translate_big = {
        'params_set': 'big',
        'vocab_file': vocab_file_v1,
        'model_dir': model_dir_v1}

# args for base
args_translate_base = {
        'params_set': 'base',
        'vocab_file': vocab_file_v0,
        'model_dir': model_dir_v0,
        }


# Train file number
_DEFAULT_TRAIN = 100
_SMALL_TRAIN = 30



def usage_test(name: str='') -> None:
    basic_usage(name)


def pre_data(corpus: str, data_dir=processed_data, shuffle=True) -> None:
    print('Run data preprocess for corpus {}'.format(corpus))
    if corpus == 'umcorpus':
        process(True, umcorpus_raw, eval_dir, data_dir, 'umcorpus', shuffle, False, _DEFAULT_TRAIN)
    elif corpus == 'multiun':
        process(False, multiun_raw, None, data_dir, 'multiun', shuffle, False, _DEFAULT_TRAIN)
    elif corpus == 'opensubtitles':
        process(False, opensubtitles_raw, None, data_dir, 'opensubtitles', shuffle, False, _DEFAULT_TRAIN)
    elif corpus == 'wmtcasia15':
        process(False, wmtcasia15_raw, None, data_dir, 'wmtcasia15', shuffle, False, _SMALL_TRAIN)
    elif corpus == 'wmtcasict15':
        process(False, wmtcasict15_raw, None, data_dir, 'wmtcasict15', shuffle, True, _SMALL_TRAIN)

<<<<<<< HEAD
def train(bleu_source=bleu_source, bleu_ref=bleu_source, num_gpus=2, params_set='base', 
        data_dir=None, model_dir=None, 
=======
def train(bleu_source=bleu_source, bleu_ref=bleu_ref, num_gpus=2, params_set='big', 
        data_dir=processed_data, model_dir=model_dir_v1, 
>>>>>>> a053b93843ade1f7c6063f4673bb67325d445e2d
        export_dir=export_dir, batch_size=None, allow_ffn_pad=True, 
        hooks=HOOKS, stop_threshold=0.15, vocab_file=None):
    print('Begin to train and eval Transformer model...')
    
    if params_set == 'base':
        data_dir = umcorpus_data
        model_dir = model_dir_v0
        vocab_file = vocab_file_v0
    else:
        data_dir = processed_data
        model_dir = model_dir_v1
        vocab_file = vocab_file_v1

    run_transformaer(num_gpus=num_gpus, params_set=params_set, data_dir=data_dir, 
            model_dir=model_dir, export_dir=export_dir, batch_size=batch_size, 
            allow_ffn_pad=allow_ffn_pad, bleu_source=bleu_source, bleu_ref=bleu_ref, 
        hooks=hooks, stop_threshold=stop_threshold, vocab_file=vocab_file)


def translate(text=None, inputs_file=None, output_file=None, args='base'):
    if args == 'base':
        args = args_translate_base
    else:
        args = args_translate_big
    translate_main(text, inputs_file, output_file, args)


def params_num(ckpt_dir):
    num = get_params_num(ckpt_dir)
    tf.logging.info('\n\nThe number of all parameters is: {}'.format(num))
    tf.logging.info('Total size: {} byte'.format(8*num))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    fire.Fire()
