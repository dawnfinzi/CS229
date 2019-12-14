"""
Loss functions for imageopt

Eshed Margalit & Dawn Finzi
"""

import tensorflow as tf

def total_variation_loss(image_tensor):
    """
    Total variation loss (TV)
    Higher TV indicates bigger differences between adjacent pixels
    """

    X = tf.squeeze(image_tensor)
    xdiff = X[1:, :, :] - X[:-1, :, :]
    ydiff = X[:, 1:, :] - X[:, :-1, :]

    xdiff_l2 = tf.sqrt(tf.reduce_sum(tf.square(xdiff)))
    ydiff_l2 = tf.sqrt(tf.reduce_sum(tf.square(ydiff)))

    tv_loss = xdiff_l2 + ydiff_l2
    return tv_loss