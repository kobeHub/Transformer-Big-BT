"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Thu 02 May 2019

 Train and evaluate the Trnasformer model.
 Using customized Estimator to define mode_fn and train eval
 behaviors.

 There are 3 mode in the Estimator:
    tf.estimator.ModeKeys.TRAIN
    tf.estimator.ModeKeys.EVAL
    tf.estimator.ModeKeys.PREDICT

 The `model_fn` must provide code to process the 3 `mode` cases and return
 a `tf.estimator.EstimatorSpec` instance.
 Estimator will call `model_fn` while user call `train`, `evaluate`,
 or `predict` methods.
"""

import os
import tempfile

import tensorflow as tf

import compute_bleu
import metrics
import controler
import translate

from model import transformer
from model import params
from preprocess import tokenizer
from preprocess import dataset



# quick use of constant
_TRAIN = tf.estimator.ModeKeys.TRAIN
_EVAL = tf.estimator.ModeKeys.EVAL
_PREDICT = tf.estimator.ModeKeys.PREDICT

DEFAULT_TRAIN_EPOCHES = 10
INF = int(1e9)
BLEU_DIR = 'bleu'



def model_fn(features, labels, mode, params):
    """Defines the model_fn, specify how to predict, evaluate, and train"""
    with tf.variable_scope('model'):
        inputs, targets = features, labels

        model = transformer.Transformer(params, mode == _TRAIN)
        logits = model(inputs, targets)

        # Prediction mode, labels are None, output is predictions
        if mode == _PREDICT:
            return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=logits,
                    export_output={
                        "translate": tf.estimator.export.PredictOutput(logits)
                        })

        logits.set_shape(targets.shape.as_list() + logits.shape.as_list()[2:])

        # Calculate the model loss
        xentropy, weights = metrics.padded_cross_entropy_loss(
                logits, targets, params['label_smoothing'], params['vocab_size'])

        # Name the tensor
        tf.identity(loss, 'cross_entropy')

        if mode == _EVAL:
            return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    predictions={'predictions': logits},
                    eval_metric_ops=metrics.get_eval_metrics(logits, labels, params))
        else:
            train_ops, metric_dict = get_train_op_and_metrics(loss, parmas)

            metric_dict['minibatch_loss'] = loss
            record_scalars(metric_dict)
            return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_ops=train_ops)


def record_scalars(metric_dict):
    for k, v in metric_dict.items():
        tf.contrib.summary.scalar(name=k, tensor=v)


def get_learning_rate(lr, hidden_size, lr_warmup_steps):
    """Calculate lr with linear warmup and rsrqt decay."""
    with tf.name_scope('learning_rate'):
        warmup_steps = tf.cast(lr_warmup_steps, tf.float32)
        step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)

        lr *= (hidden_size ** -0.5)
        # Linear warmup
        lr *= tf.minimum(1., step / warmup_steps)
        # rsqrt deacy
        lr *= tf.rsqrt(tf.maximum(step, warmup_steps))

        # name model/get_train_op/learning_rate/lr
        tf.identity(lr, 'lr')

        return lr


def get_train_op_and_metrics(loss, params):
    """Generate training ops and metrics and save into summary."""
    with tf.variable_scope('get_train_op'):
        lr = get_learning_rate(
                params['learning_rate'],
                params['hidden_size'],
                params=['learning_rate_warmup_steps'])

        # Use adamoptimizer from tf.contrib which is faster
        optimizer = tf.contrib.opt.LazyAdamOptimizer(
                lr,
                beta1=params['optimizer_adam_beta1'],
                beta2=params['optimizer_adam_beta2'],
                epsilon=params['optimizer_adam_epsilon'])

        # Get and apply gradients
        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        gradients = optimizer.compute_gradients(
                loss, tvars, colocate_gradients_with_ops=True)
        minimize_op = optimizer.apply_gradients(
                gradients, global_step=global_step, name='Train')
        update_op = tf.get_collection(tf.GraphKeys.UP_DATEOPS)
        train_ops = tf.group(minimum_op, update_ops)

        train_metrics = {'learning_rate': lr}

        gradient_norm = tf.global_norm(list(zip(*gradients))[0])
        train_metrics["global_norm/gradient_norm"] = gradient_norm

        return train_op, train_metrics


def get_gloabl_step(estimator):
    """The last ckeckpoint"""
    return int(estimator.lastest_checkpoint().split('-')[-1])


def translate_and_compute_bleu(estimator, tokenizer, bleu_source, bleu_ref):
    tmp = tempfile.NamedTemporaryFile(delete=False) 
    tmp_name = tmp.name

    # //todo
    translate.translate_file(
            estimator, tokenizer, bleu_source, output_file=tmp_name,
            print_all_translations=False)

    uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
    cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
    os.remove(tmp_filename)
    return uncased_score, cased_score


def evaluate_and_log_bleu(estimator, bleu_source, bleu_ref, vocab_file):
    tokenizer = tokenizer.Tokenizer(vocab_file)

    uncased_score, cased_score = translate_and_compute_bleu(
            estimator, tokenizer, bleu_source, bleu_ref)

    tf.logging.info('Cased bleu score: {}'.format(cased_score))
    tf.logging.info('Uncased bleu score: {}'.format(uncased_score))

    return uncased_score, cased_score


def _validate_file(filepath):
  """Make sure that file exists."""
  if not tf.gfile.Exists(filepath):
    raise tf.errors.NotFoundError(None, None, "File %s not found." % filepath)


def run_loop(estimator, controler_, train_hooks=None, bleu_source=None, 
        bleu_ref=None, bleu_threshold=None, vocab_file=None):
    """Train and evaluate loop"""
    if bleu_source:
        _validate_file(bleu_source)
    if bleu_ref:
        _validate_file(bleu_ref)
    if vocab_file:
        _validate_file(vocab_file)

    evaluate_bleu = bleu_source is not None and bleu_ref is not None
    
    # Log train detail
    tf.logging.info('Train schedule:')
    tf.logging.info('\t1. train for {}..'.format(
        controler_.train_increment_str))
    tf.logging.info('\t2. evaluate model..')
    if evaluate_bleu:
        tf.logging.info('\t3. compute BLEU..')
        if bleu_threshold:
            tf.logging.info('Repeated above untiil the BLEU score reaches {}'.format(
                bleu_threshold))
    elif not bleu_threshold:
        tf.logging.info('Repeated for {} times...'.format(controler_.train_eval_iterations))

    if evaluate_bleu:
        # Create summary writer to log bleu score
        writer = tf.summary.FileWriter(os.path.join(estimator.model_dir,
            BLEU_DIR))
    if bleu_threshold:
        controler_.train_eval_iterations = INF

    for i in range(controler_.train_eval_iterations):
        tf.logging.info('Iteration {}:'.format(i+1))
        
        # //todo
        estimator.train(dataset.train_input_fn,
                steps=controler_.single_iteration_eval_steps,
                hooks=train_hooks)

        eval_results = estimator.evaluate(input_fn=dataset.eval_input_fn,
                steps=controler_.single_iteration_eval_steps)

        tf.logging.info('Evalution results (iter {}/{})'.format(
            i+1, controler_.train_eval_iterations))
        tf.logging.info(eval_results)

        # predict
        if evaluate_bleu:
            uncased_score, cased_score = evaluate_and_log_bleu(
                    estimator,
                    bleu_source,
                    bleu_ref,
                    vocab_file)

            # Write actual bleu scores using summary writer
            global_step = get_global_step(estimator)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="bleu/uncased", simple_value=uncased_score),
                tf.Summary.Value(tag="bleu/cased", simple_value=cased_score),
            ])
            writer.add_summary(summary, global_step)
            writer.flush()

            # Stop train condition
            if uncased_score >= bleu_threshold:
                tf.logging.info('The BLEU score just passed the thresholds.')
                writer.close()
                break



def construct_estimator(model_dir, params, controler_):
    # todo 
    distribution_strategy = distribution_utils.get_distribution_strategy(
            )
    return tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            params=params,
            config=tf.estimator.RunConfig(train_distribution=distribution_strategy))


def run_transformaer(num_gpus, params_set, data_dir, model_dir, export_dir,
        num_parallel_calls, static_batch, batch_size, allow_ffn_pad, 
        use_synthetic_data, bleu_source, bleu_ref, stop_threshold, vocab_file):
    """Run the transformer train and evaluation.

    Args:
        num_gpus: num of gpus
        params_set: `base` or `tini`
        data_dir: specific the dataset directory
        model_dir: dir of the model
        num_parallel_calls: parallel calls num
        batch_size: batch_size
        allow_ffn_pad: bool
    """
    if params_set == 'base':
        params_ = params.BASE_PARAMS
    else:
        params_ = params.TINY_PARAMS

    params_['data_dir'] = data_dir
    params_['model_dir'] = model_dir
    params_['num_parallel_calls'] = num_parallel_calls
    params_['static_batch'] = static_batch
    params_['allow_ffn_pad'] = allow_ffn_pad
    params_['use_synthetic_data'] = use_synthetic_data

    # Set batch_size depends on the GPU availability
    params_['batch_size'] = batch_size or params['default_batch_size']
    params_['batch_size'] = distribution_utils.per_replica_batch_size(
            params_['batch_size'], num_gpus)

    controler_manager = controler.Controler(
            train_steps=params_['train_steps'], 
            steps_between_evals=params_['steps_between_evals'], 
            train_epoches=params_['train_epoches'],
            epoches_between_evals=params_['epoches_between_evals'], 
            default_train_epoches=DEFAULT_TRAIN_EPOCHES, 
            batch_size=params_['batch_size'],
            max_length=params_['max_length'])

    params_['repeated_dataset'] = controler_manager.repeated_dataset

    # Create hooks
    train_hooks = hook_helper.get_train_hooks(
            hooks,
            model_dir=model_dir,
            tensors_to_log=TENSORS_TO_LOG,
            batch_size=controler_manager.batch_size,
            use_tpu=False)

    # Create hooks evaluate transformer model  # todo
    estimator = construct_estimator(model_dir, params_, controler_manager)
    # Run loop
    run_loop(
            estimator=estimator, 
            controler_=controler_manager, 
            train_hooks=train_hooks, 
            bleu_source=bleu_source, 
            bleu_ref=bleu_ref, 
            bleu_threshold=stop_threshold, 
            vocab_file=vocab_file)

    # export 
    if export_dir:
        serving_input_fn = export.build_tensor_serving_input_receive_fn(
                shape=[None], dtype=tf.float64, batch_size=None)
        estimator.export_savedmodel(
                export_dir, 
                serving_input_fn,
                assets_extra={'vocab.txt': vocab_file},
                strip_default_attrs=True)

