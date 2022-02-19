from typing import Tuple
import numpy as np


def sigmoid(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.maximum(0, Z)
    cache = Z

    return A, cache

def sigmoid_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ

def relu_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    return dZ

def calc_accuracy(y_predict: np.ndarray, y: np.ndarray, m: int) -> float:
    return np.sum((y_predict == y) / m)