"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Thu 02 May 2019 03:30:13 PM CST

 Define the model parameters
"""

from collections import defaultdict

BASE_PARAMS = defaultdict(
        lambda: None,  # default value is None
        
        # input params
        default_batch_size=512,  # 512
        max_length=256,

        # model params
        initializer_gain=1.0,
        vocab_size=610390,  # //NOT FINAL
        hidden_size=256,    # 512
        num_hidden_layers=6,   # 6
        num_heads=8,
        filter_size=1024,   # ffn layer dimension 2048

        # dropout value
        layers_postprocess_dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1,

        # train  
        label_smoothing=0.1,
        learning_rate=2.0,
        learning_rate_deacy_rate=1.0,
        learning_rate_warmup_steps=16000,
        train_steps=100000,
        steps_between_evals=5000,
        eval_step=25,
        train_epoches=None,
        epoches_between_evals=1,
        

        # optimizer
        optimizer_adam_beta1=0.9,
        optimizer_adam_beta2=0.997,
        optimizer_adam_epsilon=1e-09,

        # predict
        extra_decode_length=50,
        beam_size=4,
        alpha=0.6,
        )

BIG_PARAMS = BASE_PARAMS.copy()
BIG_PARAMS.update(
        vocab_size=1132998,
        hidden_size=128,
        )


TINY_PARAMS = BASE_PARAMS.copy()
TINY_PARAMS.update(
    default_batch_size=256,
    hidden_size=32,
    num_heads=4,
    filter_size=256,
)

