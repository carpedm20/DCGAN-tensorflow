import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, batch_size, epsilon=1e-5, momentum = 0.1, name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.momentum = momentum
            self.batch_size = batch_size

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)

    def get_assigner(self):
        return self.ema.apply([self.mean, self.variance])

    def __call__(self, x, train=True):
        self.mean, self.variance = tf.nn.moments(x, [0, 1, 2])

        if train:
            shape = x.get_shape().as_list()
            
            if not shape[0] and not self.batch_size:
                raise Exception(" [!] batch_size should be specified")
            elif not shape[0]:
                shape[0] = self.batch_size

            self.mean = tf.Variable(tf.constant(0.0, shape=shape), trainable=False)
            self.variance = tf.Variable(tf.constant(1.0, shape=shape), trainable=False)

            self.gamma = tf.get_variable("gamma", shape,
                                initializer=tf.random_normal_initializer(1., 0.02))
            self.beta = tf.get_variable("beta", shape,
                                initializer=tf.constant_initializer(0.))

            if x.get_shape().ndims == 4:
                mean, variance = tf.nn.moments(x, [0, 1, 2])
            elif x.get_shape().ndims == 2:
                mean, variance = tf.nn.moments(x, [0])
            else:
                raise NotImplementedError

            assign_mean = self.mean.assign(mean)
            assign_variance = self.variance.assign(variance)
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_norm_with_global_normalization(x, self.mean,
                                                                  self.variance,
                                                                  self.beta,
                                                                  self.gamma,
                                                                  self.epsilon, True)
        else:
            mean = self.ema.average(self.mean)
            variance = self.ema.average(self.variance)

            return tf.nn.batch_norm_with_global_normalization(x, mean,
                                                              variance,
                                                              self.beta, self.gamma,
                                                              self.epsilon, True)

def binary_cross_entropy_with_logits(logits, targets, name=None):
    """Computes binary cross entropy given `logits`.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
    """
    eps = 1e-12
    with ops.op_scope([logits, targets], name, "bce_loss") as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        return -tf.reduce_mean(logits * tf.log(targets + eps) +
                               (1. - logits) * tf.log(1. - targets + eps))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis.
    """
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_dim, input_.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        return tf.nn.deconv2d(input_, w, output_shape=output_shape,
                              strides=[1, d_h, d_w, 1])

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, stddev=0.02, scope=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        return tf.matmul(input_, matrix)
