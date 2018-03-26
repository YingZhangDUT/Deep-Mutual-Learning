# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import convert_to_tfrecords
from datasets import format_market_train
from datasets import make_filename_list
from datasets.utils import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_dir', None, 'path to the raw images')
tf.app.flags.DEFINE_string('output_dir', None, 'path to the list and tfrecords ')
tf.app.flags.DEFINE_string('split_name', None, 'split name')


def main(_):

    mkdir_if_missing(FLAGS.output_dir)

    if FLAGS.split_name == 'bounding_box_train':
        format_market_train.run(image_dir=FLAGS.image_dir)

    make_filename_list.run(image_dir=FLAGS.image_dir,
                           output_dir=FLAGS.output_dir,
                           split_name=FLAGS.split_name)

    convert_to_tfrecords.run(image_dir=FLAGS.image_dir,
                             output_dir=FLAGS.output_dir,
                             split_name=FLAGS.split_name)


if __name__ == '__main__':
    tf.app.run()

