import os
import random
import numpy as np
import tensorflow as tf
from time import gmtime, strftime

from model import DCGAN
from utils import pp, save_images, to_json, make_gif, merge

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, y_dim=10,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)
        else:
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            dcgan.load(FLAGS.checkpoint_dir)

        to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
                                      [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
                                      [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
                                      [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
                                      [dcgan.h4_w, dcgan.h4_b, None])

        OPTION = 7
        if OPTION == 0:
          z_sample = np.random.uniform(-1, 1, size=(FLAGS.batch_size, dcgan.z_dim))
          save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        elif OPTION == 1:
          values = np.arange(0, 1, 1./FLAGS.batch_size)
          for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([FLAGS.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
              z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
        elif OPTION == 2:
          values = np.arange(0, 1, 1./FLAGS.batch_size)
          for idx in [random.randint(100) for _ in xrange(5)]:
            print(" [*] %d" % idx)
            z_sample = np.zeros([FLAGS.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
              z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
        elif OPTION == 3:
          values = np.arange(0, 1, 1./FLAGS.batch_size)
          for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([FLAGS.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
              z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
        elif OPTION == 4:
          image_set = []
          values = np.arange(0, 1, 1./FLAGS.batch_size)

          for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([FLAGS.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

            image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

          new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) for idx in range(64) + range(63, -1, -1)]
          make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
        elif OPTION == 5:
          image_set = []
          values = np.arange(0, 1, 1./FLAGS.batch_size)
          z_idx = [[random.randint(0,99) for _ in xrange(5)] for _ in xrange(200)]

          for idx in xrange(200):
            print(" [*] %d" % idx)
            #z_sample = np.zeros([FLAGS.batch_size, dcgan.z_dim])
            z = np.random.uniform(-1e-1, 1e-1, size=(dcgan.z_dim))
            z_sample = np.tile(z, (FLAGS.batch_size, 1))

            for kdx, z in enumerate(z_sample):
              for jdx in xrange(5):
                z_sample[kdx][z_idx[idx][jdx]] = values[kdx]

            image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

          new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 20]) for idx in range(64) + range(63, -1, -1)]
          make_gif(new_image_set, './samples/test_gif_random_merged.gif', duration=4)
        elif OPTION == 6:
          image_set = []

          z_idx = [[random.randint(0,99) for _ in xrange(10)] for _ in xrange(dcgan.z_dim)]

          for idx in xrange(100):
            print(" [*] %d" % idx)
            z = np.random.uniform(-1, 1, size=(dcgan.z_dim))
            z_sample = np.tile(z, (FLAGS.batch_size, 1))

            for kdx, z in enumerate(z_sample):
              for jdx in xrange(10):
                z_sample[kdx][z_idx[idx][jdx]] = values[kdx]

            image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
            save_images(image_set[-1], [8, 8], './samples/test_random_arange_%s.png' % (idx))

          new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) for idx in range(64) + range(63, -1, -1)]
          make_gif(new_image_set, './samples/test_gif_merged_random.gif', duration=8)
        elif OPTION == 7:
          for _ in xrange(50):
            z_idx = [[random.randint(0,99) for _ in xrange(10)] for _ in xrange(8)]

            zs = []
            for idx in xrange(8):
              z = np.random.uniform(-0.5, 0.5, size=(dcgan.z_dim))
              zs.append(np.tile(z, (8, 1)))

            z_sample = np.concatenate(zs)
            values = np.arange(0, 0.4, 0.4/8)

            for idx in xrange(FLAGS.batch_size):
              for jdx in xrange(8):
                z_sample[idx][z_idx[idx/8][jdx]] = values[idx%8]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            save_images(samples, [8, 8], './samples/multiple_testt_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))


if __name__ == '__main__':
    tf.app.run()
