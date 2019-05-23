import tensorflow as tf

from dataset import train_input_fn, eval_input_fn

import os 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data/tmp')
VOCAB_FILE = os.path.join(DATA_DIR, 'vocab.ende.610390')


class DatasetTest(tf.test.TestCase):
    
    def testTrainInputFn(self):
        params = {
                'use_synthetic_data': False,
                'batch_size': 1024,
                'max_length': 128,
                'num_parallel_calls': 4,    # The number of files to be processed comcurrently
                'repeat_dataset': 1,
                'static_batch': False,
                'data_dir': DATA_DIR,
                }
        data = train_input_fn(params)
        iterator = data.make_one_shot_iterator()
        one = iterator.get_next()
        inputs = one[0]
        target = one[1]

        l1 = tf.shape(inputs)
        l2 = tf.shape(target)

        s1 = inputs[0]
        s2 = target[0]

        with tf.Session() as sess:
            print(sess.run([l1, l2]))
            print(sess.run([s1, s2]))




if __name__ == '__main__':
    tf.test.main()



