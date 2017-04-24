import os
from glob import glob
import numpy as np
from utils import *

dataset = 'grass'
input_fname_pattern = '*.jpg'
sample_dir = 'test'
sample_num = 64 # 8x8
input_height = 60
input_width = 60
output_height = 60
output_width = 60
is_crop = False
is_grayscale = False

if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# get list of image names and shuffle them.
data = glob(os.path.join("./data", dataset, input_fname_pattern))
np.random.shuffle(data)

sample_files = data[0:sample_num]
sample = [
    get_image(sample_file,
              input_height=input_height,
              input_width=input_width,
              resize_height=output_height,
              resize_width=output_width,
              is_crop=is_crop,
              is_grayscale=is_grayscale) for sample_file in sample_files]
sample_inputs = np.array(sample).astype(np.float32)

"""
manifold_h = int(np.ceil(np.sqrt(sample_inputs.shape[0])))
manifold_w = int(np.floor(np.sqrt(sample_inputs.shape[0])))
"""
manifold_h = 2
manifold_w = 32
save_images(sample_inputs, [manifold_h, manifold_w],
            './{}/{}_data_sample.png'.format(sample_dir, dataset))
