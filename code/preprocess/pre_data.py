"""
Author: Inno Jia @ https://kobehub.github.io

The basic data preprocess. Deal with WMT18, UMcorpus, MultiUN, 
OpenSubtitles.The data format list blow:
    
    WMT18 -- name-year_ch.txt  The Chinese version of the parallel corpus.
             name-uear_en.txt  The English version of the parallel corpus.
                               String text format

    UMcorpus  -- Bi-part_name.txt  The mixed version of the English-Chinese
                                    version corpus.

    MultiUN -- MultiUN.en-zh.en English version
               MultiUN.en-zh.zh Chinese version

    OpenSubtitles -- OpenSubtitles.en-zh.en English Version
                     OpenSubtitles.en-zh.zh Chinese Version

"""

import os
import random
import time

import tensorflow as tf

from typing import Iterator, Tuple
from typing import List

from code.preprocess import tokenizer

# Define the name constants
_TRAIN_TAG = 'train'
_EVAL_TAG = 'eval'
_UMCORPUS = 'umcorpus'

# Number of the train files and eval files
_NUM_TRAIN = 100
_NUM_EVAL = 1


def text_line_iterator(path: str) -> Iterator:
    """Iterate through lines in a file"""
    with tf.gfile.Open(path) as f:
        for line in f:
            yield line



def write_file(writer, filename: str, newline=True) -> int:
    """Write all lines from file using the writer"""
    l = 0
    for line in text_line_iterator(filename):
        l += 1
        writer.write(line)
        if newline:
            writer.write('\n')
    return l


def compile_files(raw_dir: str, raw_files, name: str) -> str:
    """Combines raw files into sigle file for each language.
        UMcorpus does not need the ops.

    Args:
        raw_dir: directory of the raw files
        raw_files: dict :{'inputs': list of source language files,
                'target': list of target language files}
        name: string to append of output file

    """
    pass


def get_raw_files(raw_dir: str, excepts: List['str']=None) -> List[str]:
    abs_dir = os.path.abspath(raw_dir)
    for r, _, f in os.walk(raw_dir):
        if f:
            for item in f:
                if not excepts or item not in excepts:
                    yield os.path.join(r, item)


def split_bleu_file(test_file: str, source_path: str, target_path: str):
    """Split test file into source and target file to compute BLEU."""
    if all_exist([source_path, target_path]):
        tf.logging.info('The source file and target file are already exists.')
        return
    flag = 0
    sw = tf.gfile.Open(source_path, mode='w')
    tw = tf.gfile.Open(target_path, mode='w')
    for line in text_line_iterator(test_file):
        if flag % 2 == 0:
            sw.write(line)
        else:
            tw.write(line)
        flag = 1 - flag
    sw.close()
    tw.close()
    


def merge_umcorpus(raw_dir: str, output_file: str) -> int:
    """Merge all the train file in UMcorpus into one file."""
    tf.logging.info('Merge UMcorpus into one file {}...'.format(output_file))
    files = get_raw_files(raw_dir, ['Readme.txt', 'Testing-Data.txt'])

    lines = 0
    with tf.gfile.Open(output_file, mode='w') as writer:
        for f in files:
            tf.logging.info('\t' + f)
            lines += write_file(writer, f, False) 
    return lines



##################################################################
# Data encode
##################################################################
def encode_and_save(tokenizer, source_input, target_input, output_dir, name, tag, 
        output_nums) -> List[str]:
    """Encode data from one file in TFRecord format and save in multi files
     
      Apply to corpus:
            UMcorpus: merged file given by source_input and target_input is None
            MultiUN, OpenSubtitles, WMT18: source_input and target_input are given

        Args:
            tokenizer: Subtokenizer object to encode the raw string
            raw_files: The raw files
            output_dir: The directory to write out the examples
            name: the name of the corpus
            tag: Output files tag 
            output_nums: The num of the output files
        :r  all files that product
    """
    outputs_path = [shared_file_name(output_dir, name, tag, i+1, output_nums)
            for i in range(output_nums)]

    if all_exist(outputs_path):
        tf.logging.info('All the TFRecord files are existed in {}'.format(output_dir))
        return outputs_path
    
    tf.logging.info('Save files with tag: {}'.format(tag))
    tmp_files = [fname + '.incomplete' for fname in outputs_path]
    writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_files]
    
    counter, shared = 0, 0
    buffer_dict = {}

    if target_input is None:
        # For UMcorpus
        for counter, line in enumerate(text_line_iterator(source_input)):
            if counter > 0 and counter % 100000 == 0:
                tf.logging.info('\tSaving case %d.' % (counter/2))
            if counter % 2 == 0:
                buffer_dict['inputs'] = tokenizer.encode(line, add_eos=True)
            else:
                buffer_dict['targets'] = tokenizer.encode(line, add_eos=True)
                example = dict_to_example(buffer_dict)
                writers[shared].write(example.SerializeToString())
                shared = (shared + 1) % output_nums
    else:
        # For MultiUN
        for counter, (s_line, t_line) in enumerate(zip(
            text_line_iterator(source_input), text_line_iterator(target_input))):
            if counter > 0 and counter % 100000 == 0:
                tf.logging.info('\tSaving case {}'.format(counter))
            example = dict_to_example({'inputs': tokenizer.encode(s_line, add_eos=True),
                'targets': tokenizer.encode(t_line, add_eos=True)})
            writers[shared].write(example.SerializeToString())
            shared = (shared + 1) % output_nums

    for writer in writers:
        writer.close()

    for tmp_name, final_name in zip(tmp_files, outputs_path):
        tf.gfile.Rename(tmp_name, final_name)


    if target_input:
        tf.logging.info('Saved {} examples'.format(counter + 1))
    else:
        tf.logging.info('Saved {} examples'.format((counter + 1) // 2))
    return outputs_path
 


def shared_file_name(path, vocab, tag, num, total_num) -> str:
    """"Create name for shared files"""
    return os.path.join(path, '{}-{}-{}-of-{}'.format(
        vocab, tag, num, total_num))



def shuffle_record(filename) -> None:
    """"Shuffle record in the file"""
    tf.logging.info('Shuffle records in file {}'.format(filename))

    tmp_name = filename + '.unshuffled'
    tf.gfile.Rename(filename, tmp_name)

    reader = tf.io.tf_record_iterator(tmp_name)
    records = []
    for record in reader:
        records.append(record)
        if len(records) % 100000 == 0:
            tf.logging.info('..read: {}'.format(len(records)))

    random.shuffle(records)

    # Write into new file
    with tf.python_io.TFRecordWriter(filename) as f:
        for i, j in enumerate(records):
            f.write(j)
            if i > 0 and i % 100000 == 0:
                tf.logging.info('write: {}...'.format(i))

    tf.gfile.Remove(tmp_name)


def dict_to_example(data):
    """Convert a key-value map to tf.Example data"""
    features = {}
    for k, v in data.items():
        features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    return tf.train.Example(features=tf.train.Features(feature=features))


def safe_mkdir(path):
    if not os.path.exists(path):
        tf.logging.info('Making directory %s' % path)
        os.mkdir(path)


def all_exist(files: List[str]) -> bool:
    for f in files:
        if not os.path.exists(f):
            return False
    return True


def vocab_exist(dir_name):
    for f in os.listdir(dir_name):
        if 'vocab.ende' in f:
            return os.path.join(dir_name, f)
    return ''


def process(merged: bool, raw_dir: str, eval_dir: str, data_dir: int, 
        shuffle: bool, append_vocab: bool):
    """Get the input data for transformer model. Entry of the 2 type corpus files. 
    
    Args:
        merged: true for the UMcorpus; false for the MultiUN, OpenSubtitles, WMT18
        raw_dir: raw files 
        eval_files: None if there is no eval_files
        data_dir: the output processed data directory
        shuffle: shuffle or not
        append_vocab: append the vocabulary based on the raw_files or not
    """
    safe_mkdir(data_dir)

    tf.logging.info('Prepare data for input.....')
    # Create vocab
    tf.logging.info('\t1. Create tokenizer and build vocab..')
    start = time.time()
    train_files = list(get_raw_files(raw_dir, ['Testing-Data.txt', 'Readme.txt']))
    
    if eval_dir:
        eval_files = list(get_raw_files(eval_dir))
        vocab_files = train_files + eval_files
    else:
        vocab_files = train_files

    tf.logging.info('\t The vocab files:{}'.format(vocab_files))
    done_vocab = vocab_exist(data_dir)
    if done_vocab and not append_vocab:
        tokenizer_ = tokenizer.Tokenizer(os.path.join(data_dir, done_vocab))
    else:
        tf.logging.info('\tusing the existed vocab: {}'.format(done_vocab))
        tokenizer_ = tokenizer.Tokenizer.vocab_from_files(
            done_vocab, vocab_files, append_vocab)
    tf.logging.info('Using time {:.2f}s'.format(time.time() - start))

    if merged:
        tf.logging.info('\t2. Merge all train file into one')
        MERGED_NAME = os.path.join(data_dir, 'merged_raw.txt')
        if not os.path.exists(MERGED_NAME):
            start = time.time()
            lines = merge_umcorpus(raw_dir, MERGED_NAME)
            tf.logging.info('The merged file has {} lines'.format(lines))
            tf.logging.info('Using time {:.2f}s'.format(time.time() - start))
        else:
            tf.logging.info('The merged file already exists.')

    
        tf.logging.info('\t3. Tokenizer and save data as TFRecord format')
        start = time.time()
        train_tfrecord = encode_and_save(tokenizer_,
                MERGED_NAME, None, data_dir, name, _TRAIN_TAG, _NUM_TRAIN)
        eval_tfrecord = encode_and_save(tokenizer_,
                eval_files[0], None, data_dir, name, _EVAL_TAG, _NUM_EVAL)
        tf.logging.info('Using time {:.2f}s'.format(time.time() - start))

        tf.logging.info('\4. Split test file into source file and target file.')
        source_path = os.path.join(data_dir, 'bleu_source.txt')
        target_path = os.path.join(data_dir, 'bleu_target.txt')
        split_bleu_file(eval_files[0], source_path, target_path)
        tf.logging.info('\tWrite into {}, {}'.format(source_path, target_path))
    else:
        if len(train_files) > 2:
            tf.logging.info('The Raw files are too much, merged them')
            # todo for WMT18
        
        # after merge
        assert len(train_files) == 2
        tf.logging.info('\t2. Tokenizer and save data as TFRecord')
        start = time()
        if 'en' in train_files[0]:
            source_input = train_files[0]
            target_input = train_files[1]
        elif 'zh' in train_files[1]:
            source_input = train_files[1]
            target_input = train_files[0]
        train_tfrecord= encode_and_save(tokenizer_, 
                source_input, target_input, data_dir, name, _TRAIN_TAG, _NUM_TRAIN)
        tf.logging.info('\tUsing time {:.2f}s'.format(time.time() - start))

        

    if shuffle:
        tf.logging.info('\tLast: Shuffle the train data')
        start = time.time()
        for fname in train_tfrecord:
            shuffle_record(fname)
        tf.logging.info('Using time {}s'.format(time.time() - start))

