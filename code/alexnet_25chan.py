"""
Optimize one image each for 50 channels in Alexnet pre RELU
"""

# import packages
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse

import ipdb
import numpy as np
import tensorflow as tf

from models_alexnet import *
import transformations as xforms
from optimization import get_optimal_image

# set paths
CKPT_PATH = "/share/kalanit/Projects/Dawn/CS229/models/checkpoints/alexnet/model.ckpt-115000"
alexnet_checkpoint_path = "/share/kalanit/Projects/Dawn/CS229/models/checkpoints/alexnet/model.ckpt-115000"
SAVE_PATH = "/share/kalanit/Projects/Dawn/CS229/figures"

def main():
    print("Using GPU %s" % FLAGS.gpu)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    channels_to_plot = np.arange(25)
    fig, axes = plt.subplots(figsize=(20, 20), nrows=5, ncols=5)

    params = {
        'channel': 1,
        'learning_rate': .05,
        'regularization': 0.0001,
        'steps': 2048,
    }

    layer_names = [
        #'conv1',
        #'conv2',
        'conv3',
        'conv4',
        'conv5',
    ]

    tensor_names = [
        #'conv',
        #'conv_1',
        'conv_2',
        'conv_3',
        'conv_4',
    ]

    alexnet_kwargs = {
        'train': False
    }

    loss = 'TV'

    for layer_name, tensor_name in zip(layer_names, tensor_names):
        print("Processing %s" % layer_name)
        for channel, ax in zip(channels_to_plot, axes.ravel()):
            print("Processing channel %d" % channel)
            params['channel'] = channel
            params['tensor_name'] = tensor_name
            optimal_image = get_optimal_image(
                alexnet_no_fc_wrapper,
                alexnet_kwargs,
                alexnet_checkpoint_path,
                params,
                loss,
                layer_name=None,
            )

            ax.imshow(optimal_image)
            ax.axis('off')

        save_path = "%s/alexnet_%s_25channels.png" % (SAVE_PATH, layer_name)
        plt.savefig(save_path, dpi=300)

if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", dest="gpu", type=str, default="1", help="Which gpu to run this on"
    )

    FLAGS, _ = parser.parse_known_args()
    main()
