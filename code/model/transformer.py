"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Wed 01 May 2019 09:11:20 PM CST

 Define the Transformer model with encoder, decoder stack
"""

import tensorflow as tf
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from model import attention
from model import beam_search
from model import embedding_layers
from model import ffn
from preprocess.tokenizer import EOS_ID

_NEG_NIF = -1e9

print(EOS_ID)
