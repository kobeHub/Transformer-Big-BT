"""
 Author: Tensorflow Author
 Date: Fri 03 May 2019

 Helper functions for running models in a distributed setting.
"""


import json
import os
import random
import string
import tensorflow as tf


def _collective_communication(all_reduce_alg):
  """Return a CollectiveCommunication based on all_reduce_alg.
  Args:
    all_reduce_alg: a string specifying which collective communication to pick,
      or None.
  Returns:
    tf.distribute.experimental.CollectiveCommunication object
  Raises:
    ValueError: if `all_reduce_alg` not in [None, 'ring', 'nccl']
  """
  collective_communication_options = {
      None: tf.distribute.experimental.CollectiveCommunication.AUTO,
      "ring": tf.distribute.experimental.CollectiveCommunication.RING,
      "nccl": tf.distribute.experimental.CollectiveCommunication.NCCL
  }
  if all_reduce_alg not in collective_communication_options:
    raise ValueError(
        "When used with `multi_worker_mirrored`, valid values for "
        "all_reduce_alg are ['ring', 'nccl'].  Supplied value: {}".format(
            all_reduce_alg))
  return collective_communication_options[all_reduce_alg]


def _mirrored_cross_device_ops(all_reduce_alg, num_packs):
  """Return a CrossDeviceOps based on all_reduce_alg and num_packs.
  Args:
    all_reduce_alg: a string specifying which cross device op to pick, or None.
    num_packs: an integer specifying number of packs for the cross device op.
  Returns:
    tf.distribute.CrossDeviceOps object or None.
  Raises:
    ValueError: if `all_reduce_alg` not in [None, 'nccl', 'hierarchical_copy'].
  """
  if all_reduce_alg is None:
    return None
  mirrored_all_reduce_options = {
      "nccl": tf.distribute.NcclAllReduce,
      "hierarchical_copy": tf.distribute.HierarchicalCopyAllReduce
  }
  if all_reduce_alg not in mirrored_all_reduce_options:
    raise ValueError(
        "When used with `mirrored`, valid values for all_reduce_alg are "
        "['nccl', 'hierarchical_copy'].  Supplied value: {}".format(
            all_reduce_alg))
  cross_device_ops_class = mirrored_all_reduce_options[all_reduce_alg]
  return cross_device_ops_class(num_packs=num_packs)


def get_distribution_strategy(distribution_strategy="default",
                              num_gpus=0,
                              num_workers=1,
                              all_reduce_alg=None,
                              num_packs=1):
  """Return a DistributionStrategy for running the model.
  Args:
    distribution_strategy: a string specifying which distribution strategy to
      use. Accepted values are 'off', 'default', 'one_device', 'mirrored',
      'parameter_server', 'multi_worker_mirrored', case insensitive. 'off' means
      not to use Distribution Strategy; 'default' means to choose from
      `MirroredStrategy`, `MultiWorkerMirroredStrategy`, or `OneDeviceStrategy`
      according to the number of GPUs and number of workers.
    num_gpus: Number of GPUs to run this model.
    num_workers: Number of workers to run this model.
    all_reduce_alg: Optional. Specifies which algorithm to use when performing
      all-reduce. For `MirroredStrategy`, valid values are "nccl" and
      "hierarchical_copy". For `MultiWorkerMirroredStrategy`, valid values are
      "ring" and "nccl".  If None, DistributionStrategy will choose based on
      device topology.
    num_packs: Optional.  Sets the `num_packs` in `tf.distribute.NcclAllReduce`
      or `tf.distribute.HierarchicalCopyAllReduce` for `MirroredStrategy`.
  Returns:
    tf.distribute.DistibutionStrategy object.
  Raises:
    ValueError: if `distribution_strategy` is 'off' or 'one_device' and
      `num_gpus` is larger than 1; or `num_gpus` is negative.
  """
  if num_gpus < 0:
    raise ValueError("`num_gpus` can not be negative.")

  distribution_strategy = distribution_strategy.lower()
  if distribution_strategy == "off":
    if num_gpus > 1 or num_workers > 1:
      raise ValueError(
          "When {} GPUs and  {} workers are specified, distribution_strategy "
          "flag cannot be set to 'off'.".format(num_gpus, num_workers))
    return None

  if distribution_strategy == "multi_worker_mirrored" or num_workers > 1:
    return tf.distribute.experimental.MultiWorkerMirroredStrategy(
        communication=_collective_communication(all_reduce_alg))

  if (distribution_strategy == "one_device" or
      (distribution_strategy == "default" and num_gpus <= 1)):
    if num_gpus == 0:
      return tf.distribute.OneDeviceStrategy("device:CPU:0")
    else:
      if num_gpus > 1:
        raise ValueError("`OneDeviceStrategy` can not be used for more than "
                         "one device.")
      return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")

  if distribution_strategy in ("mirrored", "default"):
    if num_gpus == 0:
      assert distribution_strategy == "mirrored"
      devices = ["device:CPU:0"]
    else:
      devices = ["device:GPU:%d" % i for i in range(num_gpus)]
    return tf.distribute.MirroredStrategy(
        devices=devices,
        cross_device_ops=_mirrored_cross_device_ops(all_reduce_alg, num_packs))

  if distribution_strategy == "parameter_server":
    return tf.distribute.experimental.ParameterServerStrategy()

  raise ValueError(
      "Unrecognized Distribution Strategy: %r" % distribution_strategy)


def per_replica_batch_size(batch_size, num_gpus):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.
  Note that distribution strategy handles this automatically when used with
  Keras. For using with Estimator, we need to get per GPU batch.
  Args:
    batch_size: Global batch size to be divided among devices. This should be
      equal to num_gpus times the single-GPU batch_size for multi-gpu training.
    num_gpus: How many GPUs are used with DistributionStrategies.
  Returns:
    Batch size per device.
  Raises:
    ValueError: if batch_size is not divisible by number of devices
  """
  if num_gpus <= 1:
    return batch_size

  remainder = batch_size % num_gpus
  if remainder:
    err = ('When running with multiple GPUs, batch size '
           'must be a multiple of the number of available GPUs. Found {} '
           'GPUs with a batch size of {}; try --batch_size={} instead.'
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)
  return int(batch_size / num_gpus)
