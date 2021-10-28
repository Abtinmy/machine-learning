from random import random
from math import exp
import numpy as np


class Neuron:
    def __init__(self, size):
        self.weights = [random() for i in range(size + 1)]
        self.out = None
        self.delta = None
        #print(self.weights)


class Layer:
    def __init__(self, size, size_back):
        self.neurons = [Neuron(size_back) for i in range(size)]
            

class NeuralNetwork:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i - 1]) for i in range(1, len(sizes))]

    def _sigmoid(self, a):
        return 1.0 / (1.0 + exp(-a))
    
    def _derivative_sigmoid(self, o):
        return o * (1.0 - o)

    def _active(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    def forward_propagate(self, x):
        for layer in self.layers:
            new_x = []
            for neuron in layer.neurons:
                activation = self._active(neuron.weights, x)
                neuron.out = self._sigmoid(activation)
                new_x.append(neuron.out)
            x = new_x
        return x

    def back_propagate(self, y):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errors = []
            if i == len(self.layers) - 1:
                for index, neuron in enumerate(layer.neurons):
                    errors.append(neuron.out - y[index])
            else:
                    for j in range(len(layer.neurons)):
                        error = 0.0
                        for neuron in self.layers[i + 1].neurons:
                            error += (neuron.weights[j] * neuron.delta)
                        errors.append(error)
            for index, neuron in enumerate(layer.neurons):
                neuron.delta = errors[index] * self._derivative_sigmoid(neuron.out)
            
    def update(self, row, learning_rate):
        for i in range(len(self.layers)):
            x = row[:-1]
            if i != 0 :
                x = [neuron.out for neuron in self.layers[i - 1].neurons]
            for neuron in self.layers[i].neurons:
                for j in range(len(x)):
                    neuron.weights[j] -= learning_rate * neuron.delta * x[j]
                neuron.weights[j] -= learning_rate * neuron.delta
    
    def train(self, data, learning_rate, iter, type_out):
        for i in range(iter):
            error = 0.0
            for row in data:
                x = row[:-1]
                y = row[-1]
                out = self.forward_propagate(x)
                expected = [0 for j in range(type_out)]
                expected[int(row[-1])] = 1
                error += sum([(expected[j] - out[j]) ** 2 for j in range(len(expected))])
                self.back_propagate(expected)
                self.update(row, learning_rate)
            print(i, error)

    def predict(self, x):
        out = self.forward_propagate(x)
        return out.index(max(out))

