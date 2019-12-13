"""
General utils for imageopt
"""

import numpy as np

def norm_image(x):
    return (x - np.min(x))/np.ptp(x)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)