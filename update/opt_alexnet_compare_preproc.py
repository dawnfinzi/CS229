"""
Compare images optimized with and without preprocessing
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
from scipy.stats import spearmanr

from models_alexnet import *
import transformations as xforms
from opt_utils import get_optimal_image
from utils import *

# set paths
CKPT_PATH = "/share/kalanit/Projects/Dawn/CS229/models/checkpoints/alexnet/model.ckpt-115000"
SAVE_PATH = "/share/kalanit/Projects/Dawn/CS229/figures"


def main():
    print("Using GPU %s" % FLAGS.gpu)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    layer_params = {
        'conv4': {
            'channel': 0,
            'learning_rate': 0.05,
            'regularization': 0.0001,
            'steps': 500,
            'tensor_name': 'conv_3',
            'loss': 'TV',
        },
    }
    keys = [
        'conv4',
    ]

    alexnet_kwargs = {
        'train': False
    }

    loss = 'TV'

    layer_name = keys[0]
    layer_dict = layer_params[layer_name]

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
    title = "Differences from preprocessing choices" 
    fig.suptitle(title)

    optimal_image_no_preproc, loss_list = get_optimal_image(
            alexnet_no_fc_wrapper,
            alexnet_kwargs,
            CKPT_PATH,
            layer_dict,
            preproc=False,
            layer_name=None,
        )
    ax1.imshow(optimal_image_no_preproc)
    ax1.set_title("No preprocessing")

    optimal_image_preproc, loss_list = get_optimal_image(
            alexnet_no_fc_wrapper,
            alexnet_kwargs,
            CKPT_PATH,
            layer_dict,
            preproc=True,
            layer_name=None,
        )
    ax2.imshow(optimal_image_preproc)
    ax2.set_title("With preprocessing")
    save_path = "%s/alexnet_preproc_diffs_conv4.png" % (SAVE_PATH)
    plt.savefig(save_path, dpi=200)

    # now convert the images to a tensor and resize
    tf.reset_default_graph()
    # resize and stack the images
    image_tensor_no_preproc = tf.convert_to_tensor(optimal_image_no_preproc, dtype=tf.float32)
    npre = tf.image.resize_images(image_tensor_no_preproc, (224, 224))
    image_tensor_preproc = tf.convert_to_tensor(optimal_image_preproc, dtype=tf.float32)
    pre = tf.image.resize_images(image_tensor_preproc, (224, 224))
    images = tf.stack([npre,pre])
    #initialize model
    convnet = alexnet(images, train=False)
    # define output tensors of interest
    fc8_outputs = convnet.layers['fc8']
    # initialize tf Session and restore weighs
    sess = tf.Session()
    tf_saver_restore = tf.train.Saver()
    tf_saver_restore.restore(sess, CKPT_PATH)

    # run the tensors
    score = sess.run(fc8_outputs)
    winning_class = (np.argmax(score,1))
    print(winning_class)
    top_5 = np.argpartition(score, -5,axis=1)[:,-5:]
    total = len(np.ravel(top_5))
    no_repeats = len(np.unique(top_5))
    overlap = total-no_repeats
    
    print(total)
    print(overlap)
    print((overlap/total)*100) #percent overlap
    plt.figure(figsize = (20, 5))
    plt.imshow(score[:,1:100])
    save_path = "%s/alexnet_repeats_visualize_preproc_scores_conv4.png" % (SAVE_PATH)
    plt.savefig(save_path, dpi=200)

    #compute spearman's rho
    rho = spearmanr(np.argsort(score[0]),np.argsort(score[1]))
    print(rho)


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", dest="gpu", type=str, default="1", help="Which gpu to run this on"
    )

    FLAGS, _ = parser.parse_known_args()
    main()
