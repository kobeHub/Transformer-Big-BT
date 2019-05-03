"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Thu 02 May 2019

 Abstract training on a step or epoches biass.
"""

import math
import tensorflow as tf


_TRAIN, _EVAL = tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL

NUM_EXAMPLES = {
        tf.estimator.ModeKeys.TRAIN: 100000,  # todo
        tf.estimator.ModeKeys.EVAL: 3000      # todo
        }


class Controler(object):
    """Container of the abstract step and epoches basis.
    Transformer allow users to specify an epoch basis or a number of 
    steps basis.
    """
    
    def __init__(self, train_steps, steps_between_evals, train_epoches,
            epoches_between_evals, default_train_epoches, batch_size,
            max_length):
        if train_steps and train_epoches:
            raise ValueError('Both train_steps and train_epoches are defined!')

        if train_steps:
            self.train_eval_iterations = train_steps // steps_between_evals
            self._single_iteration_train_steps = steps_between_evals
            self._single_iteration_train_epoches = epoches_between_evals
        else:
            train_poches = train_epoches or default_train_epoches
            self.train_eval_iterations = train_epoches // epoches_between_evals
            self._single_iteration_train_steps = None
            self._single_iteration_train_epoches = epoches_between_evals

        self.max_length = max_length
        self.batch_size = batch_size


    @property
    def single_iteration_train_steps(self):
        return self._single_iteration_train_steps
        #return self.epoches_to_steps(
        #        num_epoches=self._single_iteration_train_steps,
        #        mode=_TRAIN)


    @property
    def single_iteration_eval_steps(self):
        return None


    @property
    def train_increment_str(self):
        if self._single_iteration_train_steps:
            return '{} steps.'.format(self._single_iteration_train_steps)
        return '{} epoches.'.format(self._single_iteration_train_epoches)


    @property
    def repeat_dataset(self):
        if (self._single_iteration_train_epoches is None and
                self._single_iteration_train_steps > NUM_EXAMPLES[_TRAIN]):
            return math.ceil(self._single_iteration_train_steps / 
                    NUM_EXAMPLES[_TRAIN])
        return self._single_iteration_train_epoches



