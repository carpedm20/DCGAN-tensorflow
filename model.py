from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir

    if self.dataset_name == 'mnist':
      self.data_X, self.data_y = self.load_mnist()
      self.c_dim = self.data_X[0].shape[-1]
    else:
      self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
      imreadImg = imread(self.data[0])
      if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
        self.c_dim = imread(self.data[0]).shape[-1]
      else:
        self.c_dim = 1

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')


    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G                  = self.generator(self.z, self.y)
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
    self.sampler            = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    #loss_type = 0 -> cross entropy
    #loss_type = 1 -> vanilla logloss
    loss_type = 0
    
    if loss_type == 0:
      #cross entropy loss
      self.d_loss_real = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
      self.d_loss_fake = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
      self.g_loss = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
      self.d_loss = self.d_loss_real + self.d_loss_fake
    elif loss_type == 1:
      #vanilla logloss
      self.d_loss_real = -tf.reduce_mean(tf.log(self.D))
      self.d_loss_fake = -tf.reduce_mean(tf.log(1-self.D_))
      self.d_loss = self.d_loss_real+self.d_loss_fake
      self.g_loss = -tf.reduce_mean(tf.log(self.D_))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def optimizer(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    return d_optim, g_optim

  def read_dataset(self, config):
    if config.dataset == 'mnist':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]
    else:
      sample_files = self.data[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)

    return sample_inputs, sample_labels

  def read_next_batch(self, config, idx):
    if config.dataset == 'mnist':
      batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
      batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
    else:
      batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
      batch = [
          get_image(batch_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for batch_file in batch_files]
      if self.grayscale:
        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
      else:
        batch_images = np.array(batch).astype(np.float32)
    return batch_images, batch_labels

  def train(self, config):
    d_optim, g_optim = self.optimizer(config)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs/{}".format(int(time.time())), self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))

    sample_inputs, sample_labels = self.read_dataset(config)

    counter = 1
    start_time = time.time()

    ## Load from checkpoint
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter

    ## start training
    for epoch in xrange(config.epoch):
      if config.dataset == 'mnist':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      else:      
        self.data = glob(os.path.join(
          "./data", config.dataset, self.input_fname_pattern))
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        batch_images, batch_labels = self.read_next_batch(config, idx)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        ### Training always assumes labels.
        # If the network is not conditional, discriminator and generator
        # will simply ignore the labels
        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
          feed_dict={
            self.inputs: batch_images,
            self.z: batch_z,
            self.y:batch_labels,
          })
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={
            self.z: batch_z,
            self.y:batch_labels,
          })
        self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={ self.z: batch_z, self.y:batch_labels })
        self.writer.add_summary(summary_str, counter)

        errD_fake = self.d_loss_fake.eval({
            self.z: batch_z,
            self.y:batch_labels
        })
        errD_real = self.d_loss_real.eval({
            self.inputs: batch_images,
            self.y:batch_labels
        })
        errG = self.g_loss.eval({
            self.z: batch_z,
            self.y: batch_labels
        })

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        if np.mod(counter, 100) == 1:
          try:
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y:sample_labels,
              },
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
          except:
            print("one pic error!...")

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def dcgan_discriminator(self, image, y, reuse):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      self.d_bn1 = batch_norm(name='d_bn1')
      self.d_bn2 = batch_norm(name='d_bn2')
      self.d_bn3 = batch_norm(name='d_bn3')

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

      return tf.nn.sigmoid(h4), h4

  def dcgan_cond_discriminator(self, image, y, reuse):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      self.d_bn1 = batch_norm(name='d_bn1')
      self.d_bn2 = batch_norm(name='d_bn2')

      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      x = conv_cond_concat(image, yb)

      h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
      h0 = conv_cond_concat(h0, yb)

      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
      h1 = tf.reshape(h1, [self.batch_size, -1])
      h1 = concat([h1, y], 1)

      h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
      h2 = concat([h2, y], 1)

      h3 = linear(h2, 1, 'd_h3_lin')

      return tf.nn.sigmoid(h3), h3

  def dcgan_generator(self, z, y):
    with tf.variable_scope("generator") as scope:
      self.g_bn0 = batch_norm(name='g_bn0')
      self.g_bn1 = batch_norm(name='g_bn1')
      self.g_bn2 = batch_norm(name='g_bn2')
      self.g_bn3 = batch_norm(name='g_bn3')

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      self.z_, self.h0_w, self.h0_b = linear(
          z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

      self.h0 = tf.reshape(
          self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(self.h0))

      self.h1, self.h1_w, self.h1_b = deconv2d(
          h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
      h1 = tf.nn.relu(self.g_bn1(self.h1))

      h2, self.h2_w, self.h2_b = deconv2d(
          h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3, self.h3_w, self.h3_b = deconv2d(
          h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(h3))

      h4, self.h4_w, self.h4_b = deconv2d(
          h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

      return tf.nn.tanh(h4)

  def dcgan_cond_generator(self, z, y):
    with tf.variable_scope("generator") as scope:
      self.g_bn0 = batch_norm(name='g_bn0')
      self.g_bn1 = batch_norm(name='g_bn1')
      self.g_bn2 = batch_norm(name='g_bn2')

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_h4 = int(s_h/2), int(s_h/4)
      s_w2, s_w4 = int(s_w/2), int(s_w/4)

      # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      z = concat([z, y], 1)

      h0 = tf.nn.relu(
          self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
      h0 = concat([h0, y], 1)

      h1 = tf.nn.relu(self.g_bn1(
          linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
      h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

      h1 = conv_cond_concat(h1, yb)

      h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
          [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
      h2 = conv_cond_concat(h2, yb)

      return tf.nn.sigmoid(
          deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def dcgan_sampler(self, z, y):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      self.g_bn0 = batch_norm(name='g_bn0')
      self.g_bn1 = batch_norm(name='g_bn1')
      self.g_bn2 = batch_norm(name='g_bn2')
      self.g_bn3 = batch_norm(name='g_bn3')

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      h0 = tf.reshape(
          linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
          [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0, train=False))

      h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, train=False))

      h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, train=False))

      h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, train=False))

      h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

      return tf.nn.tanh(h4)

  def dcgan_cond_sampler(self, z, y):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      self.g_bn0 = batch_norm(name='g_bn0')
      self.g_bn1 = batch_norm(name='g_bn1')
      self.g_bn2 = batch_norm(name='g_bn2')

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_h4 = int(s_h/2), int(s_h/4)
      s_w2, s_w4 = int(s_w/2), int(s_w/4)

      # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
      yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
      z = concat([z, y], 1)

      h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
      h0 = concat([h0, y], 1)

      h1 = tf.nn.relu(self.g_bn1(
          linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
      h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
      h1 = conv_cond_concat(h1, yb)

      h2 = tf.nn.relu(self.g_bn2(
          deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
      h2 = conv_cond_concat(h2, yb)

      return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def discriminator(self, image, y=None, reuse=False):
    #return self.dcgan_discriminator(image, y, reuse)
    return self.dcgan_cond_discriminator(image, y, reuse)

  def generator(self, z, y=None):
    #return self.dcgan_generator(z, y)
    return self.dcgan_cond_generator(z, y)

  def sampler(self, z, y):
    #return self.dcgan_sampler(z, y)
    return self.dcgan_cond_sampler(z, y)

  def load_mnist(self):
    data_dir = os.path.join("./data", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    return X/255.,y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

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
