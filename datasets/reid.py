"""
    Provides data given split name
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim

_FILE_PATTERN = '%s.tfrecord'

_SPLITS_NAMES = ['bounding_box_train', 'bounding_box_test', 'gt_bbox', 'query']

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and num_classes',
    'filename': 'The name of an image',
}


def get_num_examples(split_name):
    list_file = os.path.join(FLAGS.dataset_dir, '%s.txt' % split_name)
    num_examples = len(tf.gfile.FastGFile(list_file, 'r').readlines())

    return num_examples


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Get a dataset tuple.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in _SPLITS_NAMES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN

    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='png'),
        'image/label': tf.FixedLenFeature(
            [], dtype=tf.int64, default_value=-1),
        'image/filename': tf.FixedLenFeature(
            [], dtype=tf.string, default_value=''),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/label'),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    num_examples = get_num_examples(split_name)
    num_classes = FLAGS.num_classes

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=num_examples,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=num_classes)
