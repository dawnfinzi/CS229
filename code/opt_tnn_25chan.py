"""
Optimize one image each for 25 channels in TNN
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

from models_tnn import *
import transformations as xforms
from opt_utils import get_optimal_image

# set paths
CKPT_PATH = "/share/kalanit/Projects/Dawn/CS229/models/checkpoints/tnn/model.ckpt-1940300"
JSON_PATH = "/share/kalanit/Projects/Dawn/CS229/models/checkpoints/tnn/ff_128_neuralfit.json"
SAVE_PATH = "/share/kalanit/Projects/Dawn/CS229/figures"

def main():
    print("Using GPU %s" % FLAGS.gpu)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    channels_to_plot = np.arange(25)
    fig, axes = plt.subplots(figsize=(20, 20), nrows=5, ncols=5)

    preproc = False
    loss = 'TV'

    params = {
        'channel': 1,
        'learning_rate': 0.05,
        'regularization': 0.0001,
        'steps': 2048,
        'loss': loss,
    }

    layer_names = [
        'conv1',
        'conv2',
        'conv3',
        'conv4',
        'conv5',
        'conv6',
        'conv7',
        'conv8',
        'conv9',
        'conv10',
    ]

    tnn_kwargs = {
        'json_fpath': JSON_PATH,
        'batch_size': 1,
    }

    for layer_name in layer_names:
        for channel, ax in zip(channels_to_plot, axes.ravel()):
            print("Processing channel %d" % channel)
            params['channel'] = channel
            optimal_image, loss_list = get_optimal_image(
                tnn_no_fc_wrapper,
                tnn_kwargs,
                CKPT_PATH,
                params,
                preproc,
                layer_name=layer_name,
            )
            ax.imshow(optimal_image)
            ax.axis('off')

        if preproc is True:
            if loss is None:
                save_path = "%s/TNN_%s_25channels_preproc.png" % (SAVE_PATH, layer_name)
            else: 
                save_path = "%s/TNN_%s_25channels_preproc_%sloss.png" % (SAVE_PATH, layer_name, loss)
        else:
            if loss is None:
                save_path = "%s/TNN_%s_25channels_nopreproc.png" % (SAVE_PATH, layer_name)
            else: 
                save_path = "%s/TNNt_%s_25channels_nopreproc_%sloss.png" % (SAVE_PATH, layer_name, loss)

        print("Writing to %s" % save_path)
        plt.savefig(save_path, dpi=300)

if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", dest="gpu", type=str, default="1", help="Which gpu to run this on"
    )

    FLAGS, _ = parser.parse_known_args()
    main()
