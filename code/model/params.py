"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Thu 02 May 2019 03:30:13 PM CST

 Define the model parameters
"""

from collections import defaultdict

BASE_PARAMS = defaultdict(
        lambda: None,  # default value is None
        
        # input params
        default_batch_size=1024,
        max_length=256,

        # model params
        initializer_gain=1.0,
        vocab_size=33708,  # todo
        hidden_size=512,
        num_hidden_layers=6,
        num_head=8,
        filter_size=2048,   # ffn layer dimension

        # dropout value
        layer_postporcess_dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1,

        # train  //todo
        label_smoothing=0.1,
        learning_rate=2.0,
        learning_rate_deacy_rate=1.0,
        lenaring_rate_warmup_steps=16000,
        train_steps=1000000,
        steps_between_evals=10000,
        train_epoches=10,
        epoches_between_evals=2,

        # optimizer
        optimizer_adam_beta1=0.9,
        optimizer_adam_beta2=0.997,
        optimizer_adam_epslion=1e-09,

        # predict
        extra_decode_length=50,
        beam_size=4,
        alpha=0.6,
        )

TINY_PARAMS = BASE_PARAMS.copy()
TINY_PARAMS.update(
    default_batch_size=256,
    hidden_size=32,
    num_heads=4,
    filter_size=256,
)

