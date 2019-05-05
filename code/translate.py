"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Sun 05 May 2019

 Translate string or file use trained trnasformer model.
"""

import os

import tensorflow as tf

from code.preprocess import tokenizer


_DECODE_BATCH_SIZE = 32
_EXTRA_DECODE_LENGTH = 100
_BEAM_SIZE = 4
_ALPHA = 0.6



def _get_sorted_inputs(file_path: str):
    """Read from file and return the ordered contents according to
    decreasing line length.
    
    @r:
        list of contents and dict map to original index->sorted
    """
    with tf.gfile.Open(file_path) as f:
        inputs = [line.strip() for line in f.readlines()]

    inputs_len = [(i, len(it)) for i, it in enumerate(inputs)]
    sorted_len = sorted(inputs_len, lambda x: x[1], reverse=True)

    sorted_inputs = [None] * len(inputs_len)
    sorted_keys = [0] * len(inputs_len)

    for i, (index, _) in enumerate(inputs_len):
        sorted_inputs[i] = inputs[index]
        sorted_keys[index] = i
    return sorted_inputs, sorted_keys


def _encode_and_add_eos(line, tokenizer_):
    return tokenizer_.encode(line) + [tokenizer_.EOS_ID]


def _trim_and_decode(ids, tokenizer_):
    try:
        index = list(ids).index(tokenizer_.EOS_ID)
        return tokenizer_.decode(ids[:index])
    except ValueError:
        return tokenizer_.decode(ids)


def translate_file(estimator, tokenizer_, input_file, output_file=None,
        print_all=True):
    """Translate a file into target file."""
    batch_size = _DECODE_BATCH_SIZE

    sorted_inputs, sorted_keys = _get_sorted_inputs(input_file)
    num_decode_batch = (len(sorted_inputs) - 1) // batch_size + 1

    def input_gen():
        for i, line in enumerate(sorted_inputs):
            if i % batch_size == 0:
                batch_num = (i // batch_size) + 1
                tf.logging.info('Decoding batch {} of {}.'.format(batch_num, num_decode_batch))
            yield _encode_and_add_eos(line, tokenizer_)

    def input_fn():
        dataset = tf.data.Dataset.from_generator(input_gen,
                tf.int64, tf.TensorShape([None]))
        dataset = dataset.padded_batch(batch_size, [None])
        return dataset

    translations = []

    for i, predict in enumerate(estimator.predict(input_fn)):
        trans = _trim_and_decode(predict['outputs'], tokenizer_)
        translations.append(trans)

        if print_all:
            tf.logging.info('Translating:\n\tsource: {}\n\target: {}'.format(
                sorted_inputs[i], trans))

    if output_file:
        if tf.gfile.IsDirectory(output_file):
            raise ValueError('The output is a directory will not write into file.')
        tf.logging.info('Writing into {}'.format(output_file))
        with tf.gfile.Open(output_file, mode='w') as f:
            for i in sorted_keys:
                f.write('{}\n'.format(translations[i]))


def translate_text(estimator, tokenizer_, txt):
    encoded_txt = _encode_and_add_eos(txt, tokenizer_)

    def input_fn():
        ds = tf.data.Dataset.from_tensor(encoded_txt)
        ds = ds.batch(_DECODE_BATCH_SIZE)
        return ds

    predictions = estimator.predict(input_fn)
    translation = next(predictions)['outputs']
    translation = _trim_and_decode(translation, tokenizer_)
    tf.logging.info('Translating:\n\tsource: {}\n\target: {}'.format(
        txt, translation))








