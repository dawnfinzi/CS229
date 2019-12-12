# import packages
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import argparse
import ipdb
import sys
from pprint import pprint

import tensorflow as tf
import numpy as np

sys.path.append('../models/')
from facenet.src.models.inception_resnet_v1 import inception_resnet_v1
from facenet.src.facenet import load_model
from models_facenet import *

import transformations as xforms
from optimization import get_optimal_image

import importlib


# set paths
META_PATH = "/share/kalanit/Projects/Dawn/CS229/models/facenet/facenet_checkpoint/model-20180402-114759.meta"
CKPT_PATH = "/share/kalanit/Projects/Dawn/CS229/models/facenet/facenet_checkpoint/model-20180402-114759.ckpt-275"
MODEL_DIR = "/share/kalanit/Projects/Dawn/CS229/models/facenet/facenet_checkpoint/"
SAVE_PATH = "/share/kalanit/Projects/Dawn/CS229/figures"

def main():
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	# with tf.Graph().as_default():
	# 	with tf.Session() as sess:
	# 		load_model(MODEL_DIR)

	# 		print(tf.trainable_variables())
	# 		weights_tensor = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Conv2d_1a_3x3/weights:0")
	# 		weights = sess.run(weights_tensor)
	# 		print(weights.shape)
	# 		weights = np.moveaxis(weights, 3, 0)
	# 		fig, axes = plt.subplots(figsize=(24, 16), nrows=4, ncols=8)
	# 		for kernel_idx, (kernel, ax) in enumerate(zip(weights, axes.ravel())):
	# 			normed = (kernel - np.min(kernel)) / np.ptp(kernel)
	# 			ax.imshow(normed)
	# 			ax.axis('off')
	# 			ax.set_title(kernel_idx)

	# 		save_path = "%s/facenet_weights.png" % SAVE_PATH
	# 		plt.savefig(save_path, dpi=200)		

	sess = tf.Session()

	saver = tf.train.import_meta_graph(META_PATH)
	saver.restore(sess, CKPT_PATH)

	#print(tf.trainable_variables())


	weights_tensor = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Conv2d_1a_3x3/weights:0")
	weights = sess.run(weights_tensor)
	print(weights.shape)
	print(weights[0,0,0,0])

	weights = np.moveaxis(weights, 3, 0)
	fig, axes = plt.subplots(figsize=(24, 16), nrows=4, ncols=8)
	for kernel_idx, (kernel, ax) in enumerate(zip(weights, axes.ravel())):
		normed = (kernel - np.min(kernel)) / np.ptp(kernel)
		ax.imshow(normed)
		ax.axis('off')
		ax.set_title(kernel_idx)

	save_path = "%s/facenet_weights2.png" % SAVE_PATH
	plt.savefig(save_path, dpi=200)

if __name__ == "__main__":
	main()
