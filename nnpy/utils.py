import numpy as np

def softmax(array: np.ndarray, dim = -1):
    exp = np.exp(array)
    return exp / np.sum(exp, axis = dim, keepdims=True)