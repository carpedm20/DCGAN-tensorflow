import math
import tensorflow as tf

from ops import *

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x)

class DCGAN(object):
    def __init__(self, batch_size=64, load_size=96, fine_size= 64,
                 nz=100, ngf=64, ndf=64):
        self.batch_size = batch_size
        self.load_size = load_size
        self.fine_size = fine_size

        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf

        def get_generater_kernel(x, y, k_size, name):
            stddev = get_stddev(x, k_size, k_size)
            return tf.Variable(tf.random_normal([x, y], stddev=stddev), name=name)

        self.gk1 = get_generater_kernel(1, 2, 3, name='gk1')

    def generater(self):
        linear(input_, 6*6*512)

def SpatialConvolution(input, nInputPlane, nOutputPlane, kW, kH, dW=1, dH=1, padW=0, padH=0):
    return tf.nn.conv2d(input, [kH, kW, nInputPlane, nOutputPlane], [1, dH, dW, 1], padding='VALID')

def SpatialFullConvolution(input, nInputPlane, nOutputPlane, kW, kH, dW=1, dH=1, padW=0, padH=0):
    return tf.nn.conv2d(input, [kH, kW, nInputPlane, nOutputPlane], [1, dH, dW, 1], padding='SAME')

x = tf.placeholder(tf.int32, [None, ])

conv1 = SpatialFullConvolution(x, nz, ngf * 8, 4, 4)
tf.nn.batch_norm_with_global_normalization(conv1, )

conv2 = SpatialFullConvolution(conv1, ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1)
conv3 = SpatialFullConvolution(conv2, ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1)
conv4 = SpatialFullConvolution(conv3, ngf * 2, ngf, 4, 4, 2, 2, 1, 1)
conv5 = SpatialFullConvolution(conv4, ngf, nc, 4, 4, 2, 2, 1, 1)
