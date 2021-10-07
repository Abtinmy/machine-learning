import numpy as np

def feature_normalizer(x):
    for i in range(x.shape[1]):
        ones = np.ones(x.shape[0])
        mean = x[:, i].mean() * ones
        std = np.std(x[:, i])
        x[:, i] = (x[:, i] - mean) / std
    return x
    