"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Wed 15 May 2019 

 Testcases for the transformer model. 
"""

import tensorflow as tf


class TransformerTest(tf.test.Testcases):

    def test_EncoderStack(self):
        params = {'hidden_size': 512,
        'num_heads': 8,
        'num_hidden_layers': 6,
        'attention_dropout': 0.1,
        'filter_size': 2048,
        'allow_ffn_pad': True}

        encoder = transformer.EncoderStack(params, True)
        inputs = [254939, 495884, 494380,  59409, 177392, 580889, 280523, 224280,
       388662, 460449,  24656, 493847, 572291, 407836, 554070, 285057,
       596466,   8485, 183316, 275391, 421102,  34097, 410489, 601320,
            1,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0]
        print(encoder)
        