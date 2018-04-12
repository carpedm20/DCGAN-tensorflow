import tensorflow as tf
import tensorflow.contrib.slim as slim

from model import SimpleMNISTC
import numpy as np
from scipy import misc
import os

flags = tf.app.flags
flags.DEFINE_string("dataset_path", "./testdata", "The test dataset folder")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Directory name to save the checkpoints [checkpoint]")

FLAGS = flags.FLAGS

def load_test_dataset(sess, dataset_path):
    images_list = []
    label_list = []
    for fname in os.listdir(dataset_path):
        label = int(fname.split('_')[0])
        y_one_hot = np.zeros(10)
        y_one_hot[label] = 1
        label_list.append(y_one_hot)

        image = misc.imread(os.path.join(dataset_path, fname))
        image = np.reshape(image, 784)
        images_list.append(image)

    images_list = np.asarray(images_list)
    label_list = np.asarray(label_list)
    return images_list, label_list

def test_generated_data(sess, simple_classifier, dataset_path):
    imgs, labels = load_test_dataset(sess, dataset_path)
    acc_val = simple_classifier.accuracy.eval(feed_dict={simple_classifier.x: imgs, simple_classifier.y_: labels, simple_classifier.keep_prob: 1.0})
    print('test accuracy for img in %s: %g' % (dataset_path, acc_val))


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        simple_classifier = SimpleMNISTC(sess)
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        if FLAGS.train:
            simple_classifier.train()
        else:
            if not simple_classifier.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")
            test_generated_data(sess, simple_classifier, FLAGS.dataset_path)
    return

if __name__ == '__main__':
    tf.app.run()