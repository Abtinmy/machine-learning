import numpy as np
from util import feature_normalizer
from LinearModel import LinearModel
import matplotlib.pyplot as plt


class MultiVariableLinearRegression(LinearModel):

    def _compute_cost_multi(self, x, y):
        hypothesis = np.sum(self.theta * x, axis=1)
        return (np.transpose(hypothesis - y) * (hypothesis - y)).mean() / 2

    def _gradient_descent_multi(self, x, y):
        x_new = np.column_stack((np.ones(x.shape[0]), x))
        display_rate = int(self.iter / 5)
        for i in range(self.iter):
            hypothesis = np.sum(x_new * self.theta, axis=1)
            
            for j in range(x_new.shape[1]):
                self.theta[j] = self.theta[j] - self.alpha * ((hypothesis - y) * x_new[:, j]).mean()
                
            cost = self._compute_cost_multi(x_new, y)

            if (i + 1) % display_rate == 0:
                print(f"Iteration: {i}, theta: {self.theta}, cost: {cost}")
                self.plot(list(range(x.shape[0])), y, np.sum(self.theta * x_new, axis=1))

    def fit(self, x, y):
        self._gradient_descent_multi(x, y)

    def predict(self, x):
        x_new = np.column_stack((np.ones(x.shape[0]), x))
        return np.dot(x_new, self.theta)


def main():
    path_file = '..\machine-learning\LinearRegression\ex1data2.txt'

    with open(path_file, 'r') as dataset:
        data = np.loadtxt(dataset, delimiter=',')
        x = data[:, :-1]
        y = data[:, -1]

    x = feature_normalizer(x)
    
    model = MultiVariableLinearRegression(alpha=0.1, iter=1000, theta=np.zeros(x.shape[1] + 1))
    model.fit(x, y)
    

if __name__ == "__main__":
    main()