"""
Author: Inno Jia @ https://kobehub.github.io

The basic data preprocess. Deal with WMT18, UMcorpus, MultiUN, 
OpenSubtitles.The data format list blow:
    
    WMT18 -- name-year_ch.txt  The Chinese version of the parallel corpus.
             name-uear_en.txt  The English version of the parallel corpus.
                               String text format

    UMcorpus  -- Bi-part_name.txt  The mixed version of the English-Chinese
                                    version corpus.

    MultiUN -- 

    OpenSubtitles --

"""

import os
import random

import tensorflow as tf

from typing import Iterator, Tuple
from typing import List



def text_line_iterator(path: str) -> Iterator:
    """Iterate through lines in a file"""
    with tf.gfile.Open(path) as f:
        for line in f:
            yield line



def write_file(writer, filename: str) -> None:
    """Write all lines from file using the writer"""
    for line in text_line_iterator(filename):
        writer.write(line)
        writer.write('\n')



def compile_files(raw_dir, raw_files, ) -> Tuple(str, str):
    pass



##################################################################
# Data encode
##################################################################
def encode_and_save(tokenizer, raw_files, output_dir, tags, 
        output_nums) -> List[str]:
    """Encode data in TFRecord format and save in multi files
    
        Args:
            tokenizer: Subtokenizer object to encode the raw string
            raw_files: The raw files
            output_dir: The directory to write out the examples
            tags: Output files name
            output_nums: The num of the output files
        :r  all files that product
    """
    pass



def shared_file_name(path, vocab, tag, num, total_num) -> str:
    """"Create name for shared files"""
    return os.path.join(path, '{}-{}-{.5d}-of-{.5d})'.format(
        vocab, tag, num, total_num))




def shuffle_record(filename) -> None:
    """"Shuffle record in the file"""
    tf.logging.info('Shuffle records in file {}'.format(filename))

    tmp_name = filename + '.unshuffled'
    tf.gfile.Rename(filename, tmp_name)

    reader = tf.compat.v1.io.tf_record_iterator(tmp_name)
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



def dict_to_record(data):
    """Convert a key-value map to tf.Example data"""
    features = {}
    for k, v in data.items():
        features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    return tf.train.Example(features=tf.train.Features(feature=features))



def safe_nkdir(path):
    if not os.path.exists(path):
        tf.logging.info('Making directory %s' % path)
        os.mkdir(path)





