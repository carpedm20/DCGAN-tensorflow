import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs

from utils import *

class batch_norm(object):
    def __init__(self, size, beta, gamma, epsilon=1e-5, momentum = 0.1):
        with variable_scope("batch_norm"):
            self.ewma = tf.train.ExponentialMovingAverage(decay=0.99)

            self.mean, self.variance = tf.nn.moments(x, [0, 1, 2])

            self.gamma = tf.get_variable([size],
                                         initializer=tf.random_normal_initializer())
            self.beta = tf.get_variable([size],
                                        initializer=tf.constant_initializer())

    def __call__(self, x, train=True):
        if train:
            return tf.nn.batch_norm_with_global_normalization(x, self.mean, self.variance,
                                                              self.beta, self.gamma,
                                                              self.epsilon, True)
        else:
            mean = self.ewma_trainer.average(self.mean)
            variance = self.ewma_trainer.average(self.variance)

            return tf.nn.batch_norm_with_global_normalization(x, self.mean, self.variance,
                                                              self.beta, self.gamma,
                                                              self.epsilon, True)

def linear(args, output_size, bias, stddev=0.2, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = []
    for a in args:
        try:
            shapes.append(a.get_shape().as_list())
        except Exception as e:
            shapes.append(a.shape)

    is_vector = False
    for idx, shape in enumerate(shapes):
        if len(shape) != 2:
            is_vector = True
            args[idx] = tf.reshape(args[idx], [1, -1])
            total_arg_size += shape[0]
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable("Matrix", [total_arg_size, output_size],
                                 tf.random_normal_initializer(stddev=stddev))
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
            "Bias", [output_size],
            initializer=init_ops.constant_initializer(bias_start))

    if is_vector:
        return tf.reshape(res + bias_term, [-1])
    else:
        return res + bias_term
