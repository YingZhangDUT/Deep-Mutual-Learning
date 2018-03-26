"""
    Contains utilities for converting datasets.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
      values: A scalar or list of values.

    Returns:
      a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
      values: A string.

    Returns:
      a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, class_id, filename, height, width, image_format):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/label': int64_feature(class_id),
        'image/filename': bytes_feature(filename),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/format': bytes_feature(image_format),
    }))
