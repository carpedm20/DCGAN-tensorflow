import os
from glob import glob
import tensorflow as tf

from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, sess, batch_size=64, image_shape=[64, 64,3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default'):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.batch_size = batch_size
        self.image_shape = image_shape

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(batch_size, name='d_bn1')
        self.d_bn2 = batch_norm(batch_size, name='d_bn2')
        if not self.y_dim:
            self.d_bn3 = batch_norm(batch_size, name='d_bn3')

        self.g_bn0 = batch_norm(batch_size, name='g_bn0')
        self.g_bn1 = batch_norm(batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(batch_size, name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(batch_size, name='g_bn3')

        self.dataset_name = dataset_name

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [None, self.y_dim])

        self.image = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape)
        self.z = tf.placeholder(tf.float32, [None, self.z_dim])

        self.image_ = self.generator(self.z)
        self.D = self.discriminator(self.image)

        self.sampler = self.sampler(self.z)
        self.D_ = self.discriminator(self.image_, reuse=True)

        self.d_loss_real = binary_cross_entropy_with_logits(self.D,
                                                            tf.ones_like(self.D))
        self.d_loss_fake = binary_cross_entropy_with_logits(self.D_,
                                                            tf.zeros_like(self.D_))

        # log(D(x)) + log(1 - D(G(z)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        # log(D(G(z)))
        self.g_loss = binary_cross_entropy_with_logits(self.D_,
                                                       tf.ones_like(self.D_))

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()

        self.d_vars = [var for var in tf.trainable_variables() if var.name.find('d_')]
        self.g_vars = [var for var in tf.trainable_variables() if var.name.find('g_')]

    def train(self, config):
        """Train DCGAN"""
        data = glob(os.path.join("./data", config.dataset, "*.jpg"))
        #np.random.shuffle(data)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        with tf.control_dependencies([d_optim]):
            d_optim = self.d_bn_assigners
        with tf.control_dependencies([g_optim]):
            g_optim = self.g_bn_assigners

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        for epoch in xrange(config.epoch):
            counter = 0
            for idx in xrange(0, min(len(data), config.train_size), config.batch_size):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file) for batch_file in batch_files]

                z = np.random.uniform(-1, 1, size[config.batch_size, self.z_dim]) \
                             .astype(np.float32)
                image = np.array(batch).astype(np.float32)

                # Update G network: maximize log(D(G(z)))
                _, loss = sess.run([g_optim, self.g_loss], feed_dict={self.z:z})
                d_loss, D, D_ = sess.run([self.g_loss, self.D, self.D_],
                                         feed_dict={self.z: z, self.image: image})

                # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                _, loss = sess.run([d_optim, self.d_loss], feed_dict={self.z:z})
                d_loss, D, D_ = sess.run([self.d_loss, self.D, self.D_],
                                         feed_dict={self.z: z, self.image: image})

                counter += 1
                if np.mod(counter, 10) == 0:
                    samples = sess.run([self.sampler], feed_dict={z: z_sample})
                    samples = inverse_transform(samples)

    def discriminator(self, image, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if self.y_dim:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(spatial_conv(x, self.c_dim + self.y_dim))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim)))
            h1 = tf.reshape(h1, [h1.get_shape()[0], -1])
            h1 = tf.concat(1, [h1, y])

            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = tf.concat(1, [h2, y])

            y_ = sigmoid(linear(h2, 1, 'd_h3_lin'))
        else:
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            y_ = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

        if not self.y_dim:
            self.d_bn_assigners = tf.group(self.d_bn1.get_assigner(),
                                           self.d_bn2.get_assigner(),
                                           self.d_bn3.get_assigner())
        else:
            self.d_bn_assigners = tf.group(self.d_bn1.get_assigner(),
                                           self.d_bn2.get_assigner())

        return y_

    def generator(self, z, y=None):
        if self.y_dim:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(z, self.gf_dim*2*7*7, 'g_h1_lin')))
            h1 = tf.reshape(h1, [None, 7, 7, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, self.gf_dim, name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            x = tf.nn.sigmoid(deconv2d(h2, self.c_dim, name='g_h3'))
        else:
            # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim*8*4*4, 'g_h0_lin'),
                            [-1, 4, 4, self.gf_dim * 8]) # 4 x 4 x 1024
            h0 = tf.nn.relu(self.g_bn0(h0))

            h1 = deconv2d(h0, [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2 = deconv2d(h1, [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = deconv2d(h2, [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            x = deconv2d(h3, [self.batch_size, 64, 64, 3], name='g_h4')

        if self.y_dim:
            self.g_bn_assigners = tf.group(self.g_bn0.get_assigner(),
                                           self.g_bn1.get_assigner(),
                                           self.g_bn2.get_assigner())
        else:
            self.g_bn_assigners = tf.group(self.g_bn0.get_assigner(),
                                           self.g_bn1.get_assigner(),
                                           self.g_bn2.get_assigner(),
                                           self.g_bn3.get_assigner())

        return x

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        if self.y_dim:
            yb = tf.reshape(y, [None, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(z, self.gf_dim*2*7*7, 'g_h1_lin')))
            h1 = tf.reshape(h1, [None, 7, 7, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.bn2(deconv2d(h1, self.gf_dim, name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, self.c_dim, name='g_h3'))
        else:
            # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim*8*4*4, 'g_h0_lin'),
                            [-1, 4, 4, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, 16, 16, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, 64, 64, 3], name='g_h4')

            return tf.nn.tanh(h4)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self._max_length)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, file_name),
                        global_step = step.astype(int),
                        latest_filename = '%s_checkpoint' % self.dataset_name)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self._max_length)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)
