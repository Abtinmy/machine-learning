from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class LinearModel(ABC):
    def __init__(self, alpha=0.1, iter=100, conv=1e-5, theta=None):
        self.alpha = alpha
        self.iter = iter
        self.conv = conv
        self.theta = theta

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def plot(self, x, y_data, y_predict):
        plt.plot(x, y_data, '.', x, y_predict, '-')
        plt.show()