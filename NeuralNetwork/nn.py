from typing import Tuple
import numpy as np
from enum import Enum, auto
import utils
import h5py


class Activation(Enum):
    sigmoid = auto()
    relu = auto()


class NeuralNetwork:
    def __init__(self, dims: list) -> None:
        np.random.seed(3)

        self.num_layers = len(dims) - 1
        self.parameters = {}

        for layer in range(1, self.num_layers + 1):
            self.parameters['W' + str(layer)] = np.random.randn(dims[layer], dims[layer - 1]) * 0.01
            self.parameters['b' + str(layer)] = np.zeros((dims[layer], 1))

    def _linear_forward(self, A: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def _linear_activation_froward(self, A_prev: np.ndarray, W: np.ndarray,
                                   b: np.ndarray, activation: Activation) -> Tuple[np.ndarray, Tuple]:
        Z, linear_cache = self._linear_forward(A_prev, W, b)
        if activation == Activation.sigmoid:
            A, activation_cache = utils.sigmoid(Z)
        elif activation == Activation.relu:
            A, activation_cache = utils.relu(Z)
        else:
            raise ValueError('invalid activation function.')
        
        cache = (linear_cache, activation_cache)
        
        return A, cache
    
    def feed_forward(self, X: np.ndarray) -> Tuple[np.ndarray, list]:
        caches = []
        A = X

        for layer in range(1, self.num_layers):
            A_prev = A
            A, cache = self._linear_activation_froward(A_prev, self.parameters['W' + str(layer)],
                                                       self.parameters['b' + str(layer)], Activation.relu)
            caches.append(cache)

        Yhat, cache = self._linear_activation_froward(A, self.parameters['W' + str(self.num_layers)],
                                                       self.parameters['b' + str(self.num_layers)], Activation.sigmoid)
        caches.append(cache)

        return Yhat, caches

    def _compute_cost(self, Yhat: np.ndarray, y: np.ndarray) -> float:
        m = y.shape[1]
        cost = -(1 / m) * (np.dot(y, np.log(Yhat).T) + np.dot((1 - y), np.log(1 - Yhat).T))
        cost = np.squeeze(cost)

        return cost

    def _linear_backward(self, dZ: np.ndarray, cache: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def _linear_activation_backward(self, dA: np.ndarray, cache: tuple,
                                    activation: Activation) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        linear_cache, activation_cache = cache

        if activation == Activation.sigmoid:
            dZ = utils.sigmoid_backward(dA, activation_cache)
        elif activation == Activation.relu:
            dZ = utils.relu_backward(dA, activation_cache)
        else:
            raise ValueError('invalid activation function.')
        
        dA_prev, dW, db = self._linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def back_propagation(self, Yhat: np.ndarray, y: np.ndarray, caches: list) -> dict:
        grads = {}
        m = Yhat.shape[1]
        y = y.reshape(Yhat.shape)

        dYhat = - (np.divide(y, Yhat) - np.divide(1 - y, 1 - Yhat))

        cache = caches[self.num_layers - 1]
        
        grads['dA' + str(self.num_layers - 1)], grads['dW' + str(self.num_layers)], grads['db' + str(self.num_layers)] = self._linear_activation_backward(dYhat, cache, Activation.sigmoid)

        for layer in reversed(range(self.num_layers - 1)):
            cache = caches[layer]
            grads['dA' + str(layer)], grads['dW' + str(layer + 1)], grads['db' + str(layer + 1)] = self._linear_activation_backward(grads['dA' + str(layer + 1)],
                                                                            cache, Activation.relu)

        return grads

    def update_parameters(self, grads: dict, learning_rate: float) -> None:
        for layer in range(self.num_layers):
            self.parameters["W" + str(layer + 1)] = self.parameters["W" + str(layer + 1)] - learning_rate * grads["dW" + str(layer + 1)]
            self.parameters["b" + str(layer + 1)] = self.parameters["b" + str(layer + 1)] - learning_rate * grads["db" + str(layer + 1)]

    def train(self, X: np.ndarray, y: np.ndarray, learning_rate: float = .0075,
              iterations: int = 3000, print_cost: bool = False) -> list:
        costs = []
        for i in range(iterations):
            Yhat, caches = self.feed_forward(X)
            
            cost = self._compute_cost(Yhat, y)

            grads = self.back_propagation(Yhat, y, caches)

            self.update_parameters(grads, learning_rate)

            if print_cost and i % 100 == 0:
                print(f'iteration {i}: cost = {cost}')
                costs.append(cost)

        return costs

    def predict(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[1]
        res = np.zeros((1, m))

        probs, caches = self.feed_forward(X)

        for i in range(probs.shape[1]):
            if probs[0, i] > 0.5:
                res[0, i] = 1
            else:
                res[0, i] = 0
    
        return res


def main():
    train = h5py.File('NeuralNetwork/train_catvnoncat.h5', "r")
    train_x = np.array(train["train_set_x"][:]) 
    train_y = np.array(train["train_set_y"][:]) 

    test = h5py.File('NeuralNetwork/test_catvnoncat.h5', "r")
    test_x = np.array(test["test_set_x"][:]) 
    test_y = np.array(test["test_set_y"][:]) 

    train_x = train_x.reshape(train_x.shape[0], -1).T
    test_x = test_x.reshape(test_x.shape[0], -1).T

    train_x = train_x / 255
    test_x = test_x / 255

    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))

    nn = NeuralNetwork([12288, 7, 1])
    nn.train(train_x, train_y, iterations=2500, print_cost=True)
    
    pred_train = nn.predict(train_x)
    print(f'Accuracy on train dataset: {utils.calc_accuracy(pred_train, train_y, train_x.shape[1])}')

    pred_test = nn.predict(test_x)
    print(f'Accuracy on test dataset: {utils.calc_accuracy(pred_test, test_y, test_x.shape[1])}')

if __name__ == "__main__":
    main()


### inspired by Andrew Ng's deep learning course on coursera.