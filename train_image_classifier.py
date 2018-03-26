# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import train_models
from datasets.utils import *

slim = tf.contrib.slim

#########################
# Training Directories #
#########################

tf.app.flags.DEFINE_string('dataset_name', 'market1501',
                           'The name of the dataset to load.')

tf.app.flags.DEFINE_string('split_name', 'bounding_box_train',
                           'The name of the data split.')

tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                           'Directory name to save the checkpoints [checkpoint]')

tf.app.flags.DEFINE_string('log_dir', 'logs',
                           'Directory name to save the logs')


#########################
#      Model Settings   #
#########################

tf.app.flags.DEFINE_string('model_name', 'mobilenet_v1',
                           'The name of the architecture to train.')

tf.app.flags.DEFINE_string('preprocessing_name', None,
                           'The name of the preprocessing to use. If left as `None`, '
                           'then the model_name flag is used.')

tf.app.flags.DEFINE_float('weight_decay', 0.00004,
                          'The weight decay on the model weights.')

tf.app.flags.DEFINE_float('label_smoothing', 0.0,
                          'The amount of label smoothing.')

tf.app.flags.DEFINE_integer('batch_size', 16,
                            'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('max_number_of_steps', 200000,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_integer('ckpt_steps', 5000,
                            'How many steps to save checkpoints.')

tf.app.flags.DEFINE_integer('num_classes', 751,
                            'The number of classes.')

tf.app.flags.DEFINE_integer('num_networks', 2,
                            'The number of networks in DML.')

tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'The number of GPUs.')

tf.app.flags.DEFINE_float('temperature', 1.0,
                          'temperature parameter to produce soft predictions.')

#########################
# Optimization Settings #
#########################

tf.app.flags.DEFINE_string('optimizer', 'adam',
                           'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
                           '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float('learning_rate', 0.0002,
                          'Initial learning rate.')

tf.app.flags.DEFINE_float('adam_beta1', 0.5,
                          'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float('adam_beta2', 0.999,
                          'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0,
                          'Epsilon term for the optimizer.')


#########################
#   Default Settings    #
#########################
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1,
                            'Number of worker replicas.')

tf.app.flags.DEFINE_integer('num_ps_tasks', 0,
                            'The number of parameter servers. If the value is 0, then the parameters '
                            'are handled locally by the worker.')

tf.app.flags.DEFINE_integer('task', 0,
                            'Task id of the replica running the training.')

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          'The decay to use for the moving average.'
                          'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """)

tf.app.flags.DEFINE_integer('num_readers', 4,
                            'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4,
                            'The number of threads used to create the batches.')

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


FLAGS = tf.app.flags.FLAGS


def main(_):
    # create folders
    mkdir_if_missing(FLAGS.checkpoint_dir)
    mkdir_if_missing(FLAGS.log_dir)
    # training
    train_models.train()


if __name__ == '__main__':
    tf.app.run()
