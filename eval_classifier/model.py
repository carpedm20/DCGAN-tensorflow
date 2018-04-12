# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import os

class SimpleMNISTC(object):
  def __init__(self, sess, max_iter=20000):
    self.sess = sess
    self.max_iter = max_iter

    self.build_model()

    self.checkpoint_dir = './checkpoint'
    if not os.path.exists(self.checkpoint_dir):
      os.makedirs(self.checkpoint_dir)

    self.summary_path = './summary'
    if not os.path.exists(self.summary_path):
      os.makedirs(self.summary_path)
    ## Create summary
    tf.summary.scalar("Loss", self.cross_entropy)
    tf.summary.scalar("Train Accuracy", self.accuracy)
    self.summary_op = tf.summary.merge_all()
    self.saver = tf.train.Saver()
    self.graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % self.graph_location)
    self.train_writer = tf.summary.FileWriter(self.graph_location)
    self.train_writer.add_graph(tf.get_default_graph())

    self.writer = tf.summary.FileWriter(self.summary_path, graph=tf.get_default_graph())

  def build_model(self):
    # Create the model
    self.x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    self.y_ = tf.placeholder(tf.float32, [None, 10])
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
      x_image = tf.reshape(self.x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    hc1 = conv_layer('conv1', x_image, [5, 5, 1, 32], [32])
    # Pooling layer - downsamples by 2X.
    hp1 = pool_layer('pool1', hc1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    hc2 = conv_layer('conv2', hp1, [3, 3, 32, 64], [64])
    # Second pooling layer.
    hp2 = pool_layer('pool2', hc2)

    # Third layer
    hc3 = conv_layer('conv3', hp2, [3, 3, 64, 128], [128])
    hp3 = pool_layer('pool3', hc3)

    hc4 = conv_layer('conv4', hp3, [1, 1, 128, 128], [128])
    hp4 = pool_layer('pool4', hc4)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
      W_fc1 = weight_variable([2 * 2 * 128, 512])
      b_fc1 = bias_variable([512])

      h_pool2_flat = tf.reshape(hp4, [-1, 2 * 2 * 128])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
      self.keep_prob = tf.placeholder(tf.float32)
      h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
      W_fc2 = weight_variable([512, 10])
      b_fc2 = bias_variable([10])

      self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    with tf.name_scope('loss'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                              logits=self.y_conv)
    self.cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
      self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
      correct_prediction = tf.cast(correct_prediction, tf.float32)
    self.accuracy = tf.reduce_mean(correct_prediction)

  def train(self):
    self.sess.run(tf.global_variables_initializer())
    mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
    counter = 1
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter

    for i in range(self.max_iter):
      batch = mnist.train.next_batch(50)
      train_accuracy, train_summary = self.sess.run([self.accuracy, self.summary_op], feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1})
      self.writer.add_summary(train_summary, i)
      if i % 100 == 0:
        print('step %d, training accuracy %g' % (i, train_accuracy))
      self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.4})
      if (counter % 500) == 1:
        self.save(counter)
      counter += 1

    print('test accuracy %g' % self.accuracy.eval(feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0}))
    return

  def save(self, step):
    model_name = "SIMPLE.model"
    self.saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name), global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      print(" [*] Load SUCCESS")
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      print(" [!] Load failed")
      return False, 0

def conv_layer(name, x, shape_w, shape_b):
    with tf.name_scope(name):
        W = weight_variable(shape_w)
        b = bias_variable(shape_b)
        h = tf.nn.relu(conv2d(x, W) + b)
        return h

def pool_layer(name, h_conv, stride=[1, 2, 2, 1]):
    with tf.name_scope(name):
        h = max_pool_2x2(h_conv, stride)
        return h

def dropconnect(W, p):
    return tf.nn.dropout(W, keep_prob=p) * p

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x, stride):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=stride, strides=stride, padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

