"""
General utils for imageopt
"""

import numpy as np

def norm_image(x):
    return (x - np.min(x))/np.ptp(x)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(x)
    return exp/ exp.sum(0) #sums over axis representing columns