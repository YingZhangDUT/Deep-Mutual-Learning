"""
    Provides utilities to preprocess images for re-id.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim


IMAGE_HEIGHT = 160
IMAGE_WIDTH = 64


def preprocess_for_train(image,
                         output_height,
                         output_width):
    """Preprocesses the given image for training.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.

    Returns:
      A preprocessed image.
    """
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    image = tf.image.resize_images(image, [output_height, output_width])
    image = tf.image.random_flip_left_right(image)
    tf.summary.image('cropped_resized_image',
                     tf.expand_dims(image, 0))
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def preprocess_for_eval(image, output_height, output_width):
    """Preprocesses the given image for evaluation.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.

    Returns:
      A preprocessed image.
    """
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    image = tf.image.resize_images(image, [output_height, output_width])
    image.set_shape([output_height, output_width, 3])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def preprocess_image(image, output_height, output_width, is_training=False):
    """Preprocesses the given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.

    Returns:
      A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image, output_height, output_width)
    else:
        return preprocess_for_eval(image, output_height, output_width)
