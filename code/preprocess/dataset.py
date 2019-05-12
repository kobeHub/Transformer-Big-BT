"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Sat 04 May 2019

 Input pipiline for transformer model to read, filter and batch exmaples.
 The examples encoded in the TFRecord files contains in the format:
    {"input": [list of integers],
    "target": [list of integers]}
    each integer infers a word in the `vocab.encode.size` 
"""

import math
import os

import tensorflow as tf


# Buffer size for reading records from TFRecord files. Each training file is
# 4.9M
_READ_RECORD_BUFFER = 5 * 1000 * 1000

# Examples grouping constants. Defines length boundaries for each group.
_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1



def _load_records(file_name: str):
    return tf.data.TFRecordDataset(file_name, buffer_size=_READ_RECORD_BUFFER)


def _parse_exmaple(serialized_example):
    """Parse inputs and target Tensor from serialized tf.Example.
    The input exmaple and target exmaple length are variable.
    """
    data_fields = {
            'inputs': tf.VarLenFeature(tf.int64),
            'targets': tf.VarLenFeature(tf.int64)}
    parsed = tf.parse_single_example(serialized_example, data_fields)

    inputs = tf.sparse.to_dense(parsed['inputs'])
    targets = tf.sparse.to_dense(parsed['targets'])
    return inputs, targets


def _filter_max_length(exmaple, max_length=256):
    """Indicates whether the exmaple's length is lower than max"""
    return tf.logical_and(tf.size(exmaple[0]) <= max_length, 
            tf.size(exmaple[1]) <= max_length)


def _get_example_length(example):
    length = tf.maximum(tf.shape(example[0])[0], tf.shape(example[1])[0])
    return length


def _create_min_max_boundaries(max_length: int, 
        min_boundary=_MIN_BOUNDARY, scale=_BOUNDARY_SCALE):
    """Create min and max boundaries.
    @r:
        min and max boundaries list
    """
    boundaries = []
    a = min_boundary
    while a < max_length:
        boundaries.append(int(a))
        a = max(a + 1, a * scale)

    min_ = [0] + boundaries
    max_ = boundaries + [max_length+1]
    return min_, max_


def _batch_examples(dataset, batch_size, max_length):
    """Group exmaples by similar lengths and get dataset.
    Each batch of the similar-length examples are padded into same length, and
    may hava dofferent number of elements in each batch.

    Args:
        dataset: Dataset of unbatched examples
        batch_size: Max number of tokens per batch of examples
        max_length: Max number of tokens in an example input or target seq.

    @r:
        Dataset batched examples with similar lengths
    """
    buckets_min, buckets_max = _create_min_max_boundaries(max_length)

    # list of batch size for each bucket_id
    # bucket_batch_sizes[bucket_id] * buckets_max[bucket_id] <= batch_size
    bucket_batch_sizes = [batch_size // x for x in buckets_max]
    bucket_batch_sizes = tf.cast(bucket_batch_sizes, tf.int64)

    def example_to_bucket_id(example_input, example_target):
        """Get the bucket id of the example"""
        seq_length = _get_example_length((example_input, example_target))

        condition = tf.logical_and(
                tf.less_equal(buckets_min, seq_length),
                tf.less(seq_length, buckets_max))
        bucket_id = tf.reduce_min(tf.where(condition))
        return bucket_id

    def window_size(bucket_id):
        """Get number of examples to be grouped of given bucket_id"""
        return bucket_batch_sizes[bucket_id]

    def batching(bucket_id, grouped_dataset):
        bucket_batch_size = window_size(bucket_id)
        return grouped_dataset.padded_batch(bucket_batch_size, ([None], [None]))

    return dataset.apply(tf.data.experimental.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching,
        window_size=None,
        window_size_func=window_size))



def _read_and_batch_from_files(file_pattern: str, batch_size: int,
        max_length: int, num_parallel_calls: int, shuffle: bool, repeat: int,
        static_batch=False):
    """Create dataset.
    Args:
        repeat: the repeated times for dataset, None infers forever.
        static_batch: Whether the batches in the dataset should have static shapes.
            If True, the input is batched so that every batch has the
            shape [batch_size // max_length, max_length]. If False, the input is
            grouped by length, and batched so that batches may have different
            shapes [N, M], where:
                N * M <= batch_size
                M <= max_length
            In general, this setting should be False. Dynamic shapes allow the inputs
            to be grouped so that the number of padding tokens is minimized, and helps
            model training. In cases where the input shape must be static
            (e.g. running on TPU), this setting should be set to True.
    @r:
        tf.data.Dataset object
    """
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

    dataset = dataset.apply(
      tf.data.experimental.parallel_interleave(
          _load_records, sloppy=shuffle, cycle_length=num_parallel_calls))

    # Parse each tf.Example into dict
    dataset = dataset.map(_parse_exmaple, num_parallel_calls=num_parallel_calls)
    # Remove exmaples which length is larger than maximum
    dataset = dataset.filter(lambda x, y: _filter_max_length((x, y), max_length))

    if static_batch:
        dataset = dataset.apply(tf.data.Dataset.padded_batch(
            batch_size // max_length, ([max_length], [max_length])))
    else:
        dataset = _batch_examples(dataset, batch_size, max_length)

    dataset = dataset.repeat(repeat)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


#  todo 
def _generate_synthetic_data(params):
    pass


def train_input_fn(params):
    """Load and return dataset for train"""
    file_pattern = os.path.join(params['data_dir'] or '', '*train*')
    if params['use_synthetic_data']:
        return generate_synthetic_data(params)
    return _read_and_batch_from_files(
            file_pattern,
            params['batch_size'],
            params['max_length'],
            params['num_parallel_calls'],
            shuffle=True,
            repeat=params['repeat_dataset'],
            static_batch=params['static_batch'])


def eval_input_fn(params):
    """Load and return dataset for eval"""
    file_pattern = os.path.join(params["data_dir"] or "", "*dev*")
    if params["use_synthetic_data"]:
        return _generate_synthetic_data(params)
    return _read_and_batch_from_files(
        file_pattern, params["batch_size"], params["max_length"],
        params["num_parallel_calls"], shuffle=False, repeat=1,
        static_batch=params["static_batch"])
        

