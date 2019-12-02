"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import cv2
import numpy as np
import os
import time
import datetime
from time import gmtime, strftime
from six.moves import xrange
from PIL import Image

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()


def get_stddev(x, k_h, k_w): return 1 / \
    math.sqrt(k_w * k_h * x.get_shape()[-1])


def expand_path(path):
    return os.path.expanduser(os.path.expandvars(path))


def timestamp(s='%Y%m%d.%H%M%S', ts=None):
    if not ts:
        ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime(s)
    return st


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        # Reference: https://github.com/carpedm20/DCGAN-tensorflow/issues/162#issuecomment-315519747
        img_bgr = cv2.imread(path)
        # Reference: https://stackoverflow.com/a/15074748/
        img_rgb = img_bgr[..., ::-1]
        return img_rgb.astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    im = Image.fromarray(x[j:j + crop_h, i:i + crop_w])
    return np.array(im.resize([resize_h, resize_w], PIL.Image.BILINEAR))


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        im = Image.fromarray(image[j:j + crop_h, i:i + crop_w])
    return np.array(im.resize([resize_h, resize_w], PIL.Image.BILINEAR)) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth,
                      "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": [
                    '%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": [
                    '%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": [
                              '%.2f' % elem for elem in list(w)]})

                lines += """
          var layer_%s = {
            "layer_type": "fc",
            "sy": 1, "sx": 1,
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": [
                              '%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
          var layer_%s = {
            "layer_type": "deconv",
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx) + 2), 2**(int(layer_idx) + 2),
                   W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'", "").split()))


def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


def visualize(sess, dcgan, config, option, sample_dir='samples'):
    image_frame_dim = int(math.ceil(config.batch_size**.5))
    if option == 0:
        z_sample = np.random.uniform(-0.5, 0.5,
                                     size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [image_frame_dim, image_frame_dim], os.path.join(
            sample_dir, 'test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime())))
    elif option == 1:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(dcgan.z_dim):
            print(" [*] %d" % idx)
            z_sample = np.random.uniform(-1, 1,
                                         size=(config.batch_size, dcgan.z_dim))
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            if config.dataset == "mnist":
                y = np.random.choice(10, config.batch_size)
                y_one_hot = np.zeros((config.batch_size, 10))
                y_one_hot[np.arange(config.batch_size), y] = 1

                samples = sess.run(dcgan.sampler, feed_dict={
                    dcgan.z: z_sample, dcgan.y: y_one_hot})
            else:
                samples = sess.run(dcgan.sampler, feed_dict={
                    dcgan.z: z_sample})

            save_images(samples, [image_frame_dim, image_frame_dim], os.path.join(
                sample_dir, 'test_arange_%s.png' % (idx)))
    elif option == 2:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in [random.randint(0, dcgan.z_dim - 1) for _ in xrange(dcgan.z_dim)]:
            print(" [*] %d" % idx)
            z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
            z_sample = np.tile(z, (config.batch_size, 1))
            # z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            if config.dataset == "mnist":
                y = np.random.choice(10, config.batch_size)
                y_one_hot = np.zeros((config.batch_size, 10))
                y_one_hot[np.arange(config.batch_size), y] = 1

                samples = sess.run(dcgan.sampler, feed_dict={
                    dcgan.z: z_sample, dcgan.y: y_one_hot})
            else:
                samples = sess.run(dcgan.sampler, feed_dict={
                    dcgan.z: z_sample})

            try:
                make_gif(samples, './samples/test_gif_%s.gif' % (idx))
            except:
                save_images(samples, [image_frame_dim, image_frame_dim], os.path.join(
                    sample_dir, 'test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime())))
    elif option == 3:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(dcgan.z_dim):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, os.path.join(
                sample_dir, 'test_gif_%s.gif' % (idx)))
    elif option == 4:
        image_set = []
        values = np.arange(0, 1, 1. / config.batch_size)

        for idx in xrange(dcgan.z_dim):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            image_set.append(
                sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
            make_gif(image_set[-1], os.path.join(sample_dir,
                                                 'test_gif_%s.gif' % (idx)))

        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10])
                         for idx in range(64) + range(63, -1, -1)]
        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w
