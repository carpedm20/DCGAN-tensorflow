import tensorflow as tf

import numpy as np
from utils import pp
from model import DCGAN

flags = tf.app.flags
flags.DEFINE_integer("epoch", 10000, "Epoch to train [10000]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(sess, batch_size=FLAGS.batch_size, y_dim=10)
        else:
            dcgan = DCGAN(sess, batch_size=FLAGS.batch_size)

        if FLAGS.is_train:
            dcgan.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
