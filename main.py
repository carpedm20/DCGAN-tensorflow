import os
import numpy as np
import tensorflow as tf
from time import gmtime, strftime

from model import DCGAN
from utils import pp, save_images, to_json

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

        to_json("./web/js/gen_layers.js", [dcgan.h0_w, dcgan.h0_b], [dcgan.h1_w, dcgan.h1_b],
            [dcgan.h2_w, dcgan.h2_b], [dcgan.h3_w, dcgan.h3_b], [dcgan.h4_w, dcgan.h4_b])

        z_sample = np.random.uniform(-1, 1, size=(FLAGS.batch_size, dcgan.z_dim))

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))

if __name__ == '__main__':
    tf.app.run()
