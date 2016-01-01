import tensorflow as tf

from model import DCGAN
from utils import pp

flags = tf.app.flags
flags.DEFINE_integer("epoch", 10000, "Epoch to train [10000]")
flags.DEFINE_integer("train_size", 999999, "The size of train images [999999]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        FLAGS.sess = sess

        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(sess, batch_size=FLAGS.batch_size, y_dim=10)
        else:
            dcgan = DCGAN(sess, batch_size=FLAGS.batch_size)

    print(" [*] Finished")

if __name__ == '__main__':
    tf.app.run()
