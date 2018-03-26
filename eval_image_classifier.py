"""
    Generic evaluation script that evaluates a model using a given dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import eval_models
from datasets.utils import *

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('dataset_name', 'market1501',
                           'The name of the dataset to load.')

tf.app.flags.DEFINE_string('split_name', 'test',
                           'The name of the train/test split.')

tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string('checkpoint_dir', None,
                           'The directory where the model was written to or an absolute path to a '
                           'checkpoint file.')

tf.app.flags.DEFINE_string('eval_dir', 'results',
                           'Directory where the results are saved to.')

tf.app.flags.DEFINE_string('model_name', 'mobilenet_v1',
                           'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_integer('num_networks', 2,
                            'Number of Networks')

tf.app.flags.DEFINE_integer('num_classes', 751,
                            'The number of classes.')

tf.app.flags.DEFINE_integer('batch_size', 1,
                            'The number of samples in each batch.')

tf.app.flags.DEFINE_string('preprocessing_name', None,
                           'The name of the preprocessing to use. If left '
                           'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer('num_preprocessing_threads', 1,
                            'The number of threads used to create the batches.')

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          'The decay to use for the moving average.'
                          'If left as None, then moving averages are not used.')

#########################

FLAGS = tf.app.flags.FLAGS


def main(_):
    # create folders
    mkdir_if_missing(FLAGS.eval_dir)
    # test
    eval_models.evaluate()


if __name__ == '__main__':
    tf.app.run()

