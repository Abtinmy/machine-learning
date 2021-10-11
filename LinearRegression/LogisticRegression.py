import numpy as np
from LinearModel import LinearModel
from scipy.optimize import fmin_tnc
import matplotlib.pyplot as plt


class LogisticRegression(LinearModel):

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _compute_cost(self, theta, x, y):
        hypothesis = self._sigmoid(np.dot(x, theta))
        return -(1 / x.shape[0]) * np.sum(y * np.log(hypothesis) +
                                            (1 - y) * np.log(1 - hypothesis))

    def gradient(self, theta, x, y):
        hypothesis = self._sigmoid(np.dot(x, theta))
        return (1 / x.shape[0]) * np.dot(x.T, hypothesis - y)

    def fit(self, x, y):
        x_new = np.column_stack((np.ones(len(x)), x))
        params = fmin_tnc(func=self._compute_cost, x0=self.theta, fprime=self.gradient, args=(x_new, y))
        self.theta = params[0]
        return params[0]

    def predict(self, x):
        x_new = np.column_stack((np.ones(len(x)), x))
        return self._sigmoid(np.dot(x_new, self.theta))
    
    def accuracy(self, x, y, threshold=0.5):
        y_predict = (self.predict(x) >= threshold).astype(int)
        accuracy = np.mean(y == y_predict)
        return accuracy * 100

def main():
    path_file = '..\machine-learning\LinearRegression\ex2data1.txt'

    with open(path_file, 'r') as dataset:
        data = np.loadtxt(dataset, delimiter=',')
        x = data[:, :-1]
        y = data[:, -1]
        mask_ones = data[:, -1] == 1
        mask_zeroes = data[:, -1] == 0
        ones = data[mask_ones, :-1]
        zeros = data[mask_zeroes, :-1]

    plt.scatter(ones[:, 0], ones[:, 1])
    plt.scatter(zeros[:, 0], zeros[:, 1])
    #plt.show()

    model = LogisticRegression(theta=np.zeros(x.shape[1] + 1))
    theta = model.fit(x, y)
    print(theta)

    x_vals = [np.min(x[:, 1]), np.max(np.max(x[:, 1]))]
    y_vals = -(theta[0] + np.dot(theta[1], x_vals)) / theta[2]
    plt.plot(x_vals, y_vals)
    plt.show()

    print(model.accuracy(x, y))

if __name__ == "__main__":
    main()


#TODO: implemention of logistic regression using newton's method