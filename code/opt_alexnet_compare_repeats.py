"""
Compare the images optimized on different iterations
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

iterations = 25

def main():
    print("Using GPU %s" % FLAGS.gpu)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    layer_params = {
        'conv5': {
            'channel': 6,
            'learning_rate': 0.05,
            'regularization': 0.0001,
            'steps': 500,
            'tensor_name': 'conv_4',
            'loss': 'TV',
        },
    }
    keys = [
        'conv5',
    ]

    alexnet_kwargs = {
        'train': False
    }

    loss = 'TV'

    images = np.zeros((iterations, 128, 128, 3))
    layer_name = keys[0]
    layer_dict = layer_params[layer_name]

    fig, axes = plt.subplots(figsize=(20, 20), nrows=5, ncols=5)

    for its, ax in enumerate(fig.axes):
        optimal_image, loss_list = get_optimal_image(
            alexnet_no_fc_wrapper,
            alexnet_kwargs,
            CKPT_PATH,
            layer_dict,
            preproc=False,
            layer_name=None,
        )
        print(its)
        images[its,:,:,:] = optimal_image

        ax.imshow(optimal_image)
        ax.axis('off')

    save_path = "%s/alexnet_repeat_testing%d.png" % (SAVE_PATH,iterations)
    plt.savefig(save_path, dpi=200)

    # now convert the images to a tensor and resize
    tf.reset_default_graph()
    image_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    #resize for alexnet
    resized_images = tf.image.resize_images(image_tensor, (224, 224))
    #initialize model
    convnet = alexnet(resized_images, train=False)
    # define output tensors of interest
    fc8_outputs = convnet.layers['fc8']
    # initialize tf Session and restore weighs
    sess = tf.Session()
    tf_saver_restore = tf.train.Saver()
    tf_saver_restore.restore(sess, CKPT_PATH)

    # run the tensors
    logits = sess.run(fc8_outputs)
    # apply softmax to output
    probs = np.zeros((logits.shape))
    for col in range(len(logits)): #ugly but works
        probs[col,:] = softmax(logits[col,:])
    winning_class = (np.argmax(probs,1)) #top 1
    print(winning_class)
    # get top 5 and compute overlap
    top_5 = np.argpartition(probs, -5,axis=1)[:,-5:]
    total = len(np.ravel(top_5))
    no_repeats = len(np.unique(top_5))
    overlap = total-no_repeats
    print(total)
    print(overlap)
    print((overlap/total)*100) #percent overlap

    # visualize the probs across repeats
    fig,ax = plt.subplots(figsize=(20, 5))
    img = ax.imshow(logits[:,1:100])
    ax.set_xlabel('Class logits for the first 100 classes')
    ax.set_title("Class logits across repeated optimizations")
    fig.colorbar(img, ax=ax)
    save_path = "%s/alexnet_repeats_visualize_class_logits.png" % (SAVE_PATH)
    plt.savefig(save_path, dpi=200)

    #get one spearman's rho as a baseline
    baseline_rho = spearmanr(np.argsort(logits[0]),np.argsort(logits[1]))
    print(baseline_rho)


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", dest="gpu", type=str, default="1", help="Which gpu to run this on"
    )

    FLAGS, _ = parser.parse_known_args()
    main()
