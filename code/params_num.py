"""
 Author: Inno Jia @ https://kobehub.github.io
 Date: Thu 16 May 2019 05:51:12 PM CST

 Get the parameter number from init checkpoint files
"""

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import os 
import numpy as np

def get_params_num(ckpt_dir):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt.model_checkpoint_path:
        ckpt_path = ckpt.model_checkpoint_path
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    
    data=np.array([])
    var_to_shape_map=reader.get_variable_to_shape_map()

    num = 0
    for key in var_to_shape_map:
        print('tensor_name',key)
        ckpt_data=np.array(reader.get_tensor(key))#cast list to np arrary
        ckpt_data=ckpt_data.flatten()#flatten list
        num += ckpt_data.shape[0]
    return num



