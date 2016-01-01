import tensorflow as tf

from model import DCGAN
from utils import pp

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100000, "Epoch to train [100000]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        FLAGS.sess = sess

        if FLGAS.dataset == 'mnist':
            dcgan = DCGAN(y_dim=10)
        else:
            dcgan = DCGAN()

if __name__ == '__main__':
    tf.app.run()
