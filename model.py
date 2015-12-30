import tensorflow as tf

from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, batch_size=64, load_size=96, fine_size= 64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3):
        """

        Args:
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.batch_size = batch_size
        self.load_size = load_size
        self.fine_size = fine_size

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        self.z = tf.placeholder(tf.float32)

        self.discrim = self.discriminator(self.x, self.y)
        self.gen = self.generater(self.z)

    def discriminator(self, x, y):
        if y:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            x = conv_cond_concat(x, yb)

            h0 = lrelu(spatial_conv(x, self.c_dim + self.y_dim))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim)))
            h1 = tf.reshape(h1, [h1.get_shape()[0], -1])
            h1 = tf.concat(1, [h1, y])

            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim)))
            h2 = tf.concat(1, [h2, y])
            return sigmoid(linear(h2, 1))
        else:
            h0 = lrelu(conv2d(x, self.df_dim))
            h1 = lrelu(self.d_bn1(conv2d(x, self.df_dim*2)))
            h2 = lrelu(self.d_bn2(conv2d(x, self.df_dim*4)))
            h3 = lrelu(self.d_bn3(conv2d(x, self.df_dim*8)))
            return linear(tf.reshape(h3, [None, -1]), 1)

    def generater(self, z, y=None):
        if y:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(bn0(linear(z, self.gfc_dim)))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(bn1(linear(z, self.gf_dim*2*7*7)))
            h1 = tf.reshape(h1, [None, 7, 7, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(bn2(deconv2d(h1, self.gf_dim, name='h2')))
            h2 = conv_cond_concat(h2, yb)
            return tf.nn.sigmoid(deconv2d(h2, self.c_dim, name='h3'))
        else:
            h0 = tf.nn.relu(bn0(linear(z, self.gf_dim*8*4*4)))
            h0 = tf.reshape(h1, [None, 4, 4, self.gf_dim * 8])

            h1 = deconv2d(h0, self.gf_dim*4, name='h1')
            h1 = tf.relu(bn1(h1))

            h2 = deconv2d(h1, self.gf_dim*2, name='h2')
            h2 = tf.relu(bn2(h2))

            h3 = deconv2d(h2, self.gf_dim*1, name='h3')
            h3 = tf.relu(bn3(h3))

            h4 = deconv2d(h3, 3, name='h4')
            return tf.nn.tanh(h4)

    def sampler(self, z, y=None):
        if y:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(bn0(linear(z, self.gfc_dim)))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(bn1(linear(z, self.gf_dim*2*7*7)))
            h1 = tf.reshape(h1, [None, 7, 7, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(bn2(deconv2d(h1, self.gf_dim, name='h2')))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, self.c_dim, name='h3'))
        else:
            h0 = tf.nn.relu(bn0(linear(z, self.gf_dim*8*4*4)))
            h0 = tf.reshape(h1, [None, 4, 4, self.gf_dim * 8])

            h1 = deconv2d(h0, self.gf_dim*4, name='h1')
            h1 = tf.relu(bn1(h1))

            h2 = deconv2d(h1, self.gf_dim*2, name='h2')
            h2 = tf.relu(bn2(h2))

            h3 = deconv2d(h2, self.gf_dim*1, name='h3')
            h3 = tf.relu(bn3(h3))

            h4 = deconv2d(h3, 3, name='h4')
            return tf.nn.tanh(h4)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        else:
            raise Exception(" [!] Trest mode but no checkpoint found")
