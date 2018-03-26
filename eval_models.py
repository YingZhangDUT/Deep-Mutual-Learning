"""
    Generic evaluation script that evaluates a model using a given dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import math
from datetime import datetime
import numpy as np
import os.path
import sys
import scipy.io as sio

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


def _extract_once(features, labels, filenames, num_examples, saver):
    """Extract Features.
    """
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    with tf.device('/cpu:0'):
        with tf.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                if os.path.isabs(ckpt.model_checkpoint_path):
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                    saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, ckpt_name))
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Succesfully loaded model from %s at step=%s.' %
                      (ckpt.model_checkpoint_path, global_step))
            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
                # num_examples = get_num_examples()
                num_iter = int(math.ceil(num_examples / FLAGS.batch_size))
                # Counts the number of correct predictions.
                step = 0
                all_features = []
                all_labels = []
                print("Current Path: %s" % os.getcwd())
                print('%s: starting extracting features on (%s).' % (datetime.now(), FLAGS.split_name))
                while step < num_iter and not coord.should_stop():
                    step += 1
                    sys.stdout.write('\r>> Extracting %s image %d/%d' % (FLAGS.split_name, step, num_examples))
                    sys.stdout.flush()
                    eval_features, eval_labels, eval_filenames = sess.run([features, labels, filenames])
                    # print('Filename:%s, Camid:%d, Label:%d' % (eval_filenames, eval_camids, eval_labels))
                    concat_features = np.concatenate(eval_features, axis=3)
                    eval_features = np.reshape(concat_features, [concat_features.shape[0], -1])
                    all_features.append(eval_features)
                    all_labels.append(eval_labels)

                # save features and labels
                np_features = np.asarray(all_features)
                np_features = np.reshape(np_features, [len(all_features), -1])
                np_labels = np.asarray(all_labels)
                np_labels = np.reshape(np_labels, len(all_labels))
                feature_filename = "%s/%s_features.mat" % (FLAGS.eval_dir, FLAGS.split_name)
                sio.savemat(feature_filename, {'feature': np_features})
                label_filename = "%s/%s_labels.mat" % (FLAGS.eval_dir, FLAGS.split_name)
                sio.savemat(label_filename, {'label': np_labels})
                print("Done!\n")

            except Exception as e:
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.split_name, FLAGS.dataset_dir)

        ####################
        # Select the model #
        ####################
        network_fn = {}
        model_names = [net.strip() for net in FLAGS.model_name.split(',')]
        for i in range(FLAGS.num_networks):
            network_fn["{0}".format(i)] = nets_factory.get_network_fn(
                model_names[i],
                num_classes=dataset.num_classes,
                is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label, filename] = provider.get(['image', 'label', 'filename'])

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = network_fn['0'].default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels, filenames = tf.train.batch(
            [image, label, filename],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        net_endpoints, net_features = {}, {}
        all_features = []
        for i in range(FLAGS.num_networks):
            _, net_endpoints["{0}".format(i)] = network_fn["{0}".format(i)](images, scope=('dmlnet_%d' % i))
            net_features["{0}".format(i)] = net_endpoints["{0}".format(i)]['PreLogits']
            all_features.append(net_features["{0}".format(i)])

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)
        _extract_once(all_features, labels, filenames, dataset.num_samples, saver)
