"""
    Generic training script that trains a model using a given dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets.utils import *
import numpy as np

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


def _average_gradients(tower_grads, catname=None):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(input=g, axis=0)
            # print(g)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads, name=catname)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def kl_loss_compute(logits1, logits2):
    """ KL loss
    """
    pred1 = tf.nn.softmax(logits1)
    pred2 = tf.nn.softmax(logits2)
    loss = tf.reduce_mean(tf.reduce_sum(pred2 * tf.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))

    return loss


def _tower_loss(network_fn, images, labels):
    """Calculate the total loss on a single tower running the reid model."""
    # Build inference Graph.
    net_logits, net_endpoints, net_raw_loss, net_pred, net_features = {}, {}, {}, {}, {}
    for i in range(FLAGS.num_networks):
        net_logits["{0}".format(i)], net_endpoints["{0}".format(i)] = \
            network_fn["{0}".format(i)](images, scope=('dmlnet_%d' % i))
        net_raw_loss["{0}".format(i)] = tf.losses.softmax_cross_entropy(
                logits=net_logits["{0}".format(i)], onehot_labels=labels,
                label_smoothing=FLAGS.label_smoothing, weights=1.0)
        net_pred["{0}".format(i)] = net_endpoints["{0}".format(i)]['Predictions']

        if 'AuxLogits' in net_endpoints["{0}".format(i)]:
            net_raw_loss["{0}".format(i)] += tf.losses.softmax_cross_entropy(
                logits=net_endpoints["{0}".format(i)]['AuxLogits'], onehot_labels=labels,
                label_smoothing=FLAGS.label_smoothing, weights=0.4, scope='aux_loss')

    # Add KL loss if there are more than one network
    net_loss, kl_loss, net_reg_loss, net_total_loss, net_loss_averages, net_loss_averages_op = {}, {}, {}, {}, {}, {}

    for i in range(FLAGS.num_networks):
        net_loss["{0}".format(i)] = net_raw_loss["{0}".format(i)]
        for j in range(FLAGS.num_networks):
            if i != j:
                kl_loss["{0}{0}".format(i, j)] = kl_loss_compute(net_logits["{0}".format(i)], net_logits["{0}".format(j)])
                net_loss["{0}".format(i)] += kl_loss["{0}{0}".format(i, j)]
                tf.summary.scalar('kl_loss_%d%d' % (i, j), kl_loss["{0}{0}".format(i, j)])

        net_reg_loss["{0}".format(i)] = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=('dmlnet_%d' % i))
        net_total_loss["{0}".format(i)] = tf.add_n([net_loss["{0}".format(i)]] +
                                                   net_reg_loss["{0}".format(i)],
                                                   name=('net%d_total_loss' % i))

        net_loss_averages["{0}".format(i)] = tf.train.ExponentialMovingAverage(0.9, name='net%d_avg' % i)
        net_loss_averages_op["{0}".format(i)] = net_loss_averages["{0}".format(i)].apply(
            [net_loss["{0}".format(i)]] + [net_total_loss["{0}".format(i)]])

        tf.summary.scalar('net%d_loss_raw' % i, net_raw_loss["{0}".format(i)])
        tf.summary.scalar('net%d_loss_sum' % i, net_loss["{0}".format(i)])
        tf.summary.scalar('net%d_loss_avg' % i, net_loss_averages["{0}".format(i)].average(net_loss["{0}".format(i)]))

        with tf.control_dependencies([net_loss_averages_op["{0}".format(i)]]):
            net_total_loss["{0}".format(i)] = tf.identity(net_total_loss["{0}".format(i)])

    return net_total_loss, net_pred


def train():
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.split_name, FLAGS.dataset_dir)

        ######################
        # Select the network and #
        ######################
        network_fn = {}
        model_names = [net.strip() for net in FLAGS.model_name.split(',')]
        for i in range(FLAGS.num_networks):
            network_fn["{0}".format(i)] = nets_factory.get_network_fn(
                model_names[i],
                num_classes=dataset.num_classes,
                weight_decay=FLAGS.weight_decay,
                is_training=True)

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            net_opt = {}
            for i in range(FLAGS.num_networks):
                net_opt["{0}".format(i)] = tf.train.AdamOptimizer(FLAGS.learning_rate,
                                                                  beta1=FLAGS.adam_beta1,
                                                                  beta2=FLAGS.adam_beta2,
                                                                  epsilon=FLAGS.opt_epsilon)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name  # or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        with tf.device(deploy_config.inputs_device()):
            examples_per_shard = 1024
            min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=min_queue_examples + 3 * FLAGS.batch_size,
                common_queue_min=min_queue_examples)
            [image, label] = provider.get(['image', 'label'])

            train_image_size = network_fn["{0}".format(0)].default_image_size

            image = image_preprocessing_fn(image, train_image_size, train_image_size)

            images, labels = tf.train.batch(
                [image, label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=2 * FLAGS.num_preprocessing_threads * FLAGS.batch_size)
            labels = slim.one_hot_encoding(labels, dataset.num_classes)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=16 * deploy_config.num_clones,
                num_threads=FLAGS.num_preprocessing_threads)

            images, labels = batch_queue.dequeue()

            images_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=images)
            labels_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=labels)

        precision, net_tower_grads, net_update_ops,  net_var_list, net_grads = {}, {}, {}, {}, {}

        for i in range(FLAGS.num_networks):
            net_tower_grads["{0}".format(i)] = []

        for k in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % k):
                with tf.name_scope('tower_%d' % k) as scope:
                    with tf.variable_scope(tf.get_variable_scope()):

                        net_loss, net_pred = _tower_loss(network_fn, images_splits[k], labels_splits[k])

                        truth = tf.argmax(labels_splits[k], axis=1)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        var_list = tf.trainable_variables()

                        for i in range(FLAGS.num_networks):
                            predictions = tf.argmax(net_pred["{0}".format(i)], axis=1)
                            precision["{0}".format(i)] = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

                            # Add a summary to track the training precision.
                            summaries.append(tf.summary.scalar('precision_%d' % i, precision["{0}".format(i)]))

                            net_update_ops["{0}".format(i)] = \
                                tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=('%sdmlnet_%d' % (scope, i)))

                            net_var_list["{0}".format(i)] = \
                                [var for var in var_list if 'dmlnet_%d' % i in var.name]

                            net_grads["{0}".format(i)] = net_opt["{0}".format(i)].compute_gradients(
                                net_loss["{0}".format(i)], var_list=net_var_list["{0}".format(i)])

                            net_tower_grads["{0}".format(i)].append(net_grads["{0}".format(i)])

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        for i in range(FLAGS.num_networks):
            net_grads["{0}".format(i)] = _average_gradients(net_tower_grads["{0}".format(i)],
                                                            catname=('dmlnet_%d_cat' % i))

        # Add histograms for histogram and trainable variables.
        for i in range(FLAGS.num_networks):
            for grad, var in net_grads["{0}".format(i)]:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        #################################
        # Configure the moving averages #
        #################################

        if FLAGS.moving_average_decay:
            moving_average_variables = {}
            all_moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
            for i in range(FLAGS.num_networks):
                moving_average_variables["{0}".format(i)] = \
                    [var for var in all_moving_average_variables if 'dmlnet_%d' % i in var.name]
                net_update_ops["{0}".format(i)].append(
                    variable_averages.apply(moving_average_variables["{0}".format(i)]))

        # Apply the gradients to adjust the shared variables.
        net_grad_updates, net_train_op = {}, {}
        for i in range(FLAGS.num_networks):
            net_grad_updates["{0}".format(i)] = net_opt["{0}".format(i)].apply_gradients(
                net_grads["{0}".format(i)], global_step=global_step)
            net_update_ops["{0}".format(i)].append(net_grad_updates["{0}".format(i)])
            # Group all updates to into a single train op.
            net_train_op["{0}".format(i)] = tf.group(*net_update_ops["{0}".format(i)])

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(
            os.path.join(FLAGS.log_dir),
            graph=sess.graph)

        net_loss_value, precision_value = {}, {}

        for step in xrange(FLAGS.max_number_of_steps):

            for i in range(FLAGS.num_networks):
                _, net_loss_value["{0}".format(i)], precision_value["{0}".format(i)] = \
                    sess.run([net_train_op["{0}".format(i)], net_loss["{0}".format(i)],
                              precision["{0}".format(i)]])
                assert not np.isnan(net_loss_value["{0}".format(i)]), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                format_str = '%s: step %d, net0_loss = %.2f, net0_acc = %.4f'
                print(format_str % (FLAGS.dataset_name, step, net_loss_value["{0}".format(0)],
                                    precision_value["{0}".format(0)]))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % FLAGS.ckpt_steps == 0 or (step + 1) == FLAGS.max_number_of_steps:
                checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

