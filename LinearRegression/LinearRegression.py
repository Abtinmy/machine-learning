from LinearModel import LinearModel
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class LinearRegression(LinearModel):

    def _compute_cost(self, x, y):
        hypothesis = np.dot(x, self.theta)
        return (((hypothesis - y) ** 2).mean()) / 2

    def _gradient_descent(self, x, y):
        x_new = np.column_stack((np.ones(len(x)), x))
        display_rate = int(self.iter / 5)
        for i in range(self.iter):
            hypothesis = np.dot(x_new, self.theta)
            theta_0 = self.theta[0] - self.alpha * ((hypothesis - y).mean())
            theta_1 = self.theta[1] - self.alpha * (((hypothesis - y) * x_new[:, 1]).mean())

            self.theta = np.array([theta_0, theta_1])
            cost = self._compute_cost(x_new, y)

            if (i + 1) % display_rate == 0:
                print(f"Iteration: {i}, theta: {self.theta}, cost: {cost}")
                self.plot(x, y, np.dot(x_new, self.theta))

    def fit(self, x, y):
        self._gradient_descent(x, y)

    def predict(self, x):
        x_new = np.column_stack((np.ones(len(x)), x)) 
        return np.dot(x_new, self.theta)


def main():
    path_file = '..\machine-learning\LinearRegression\ex1data1.txt'

    with open(path_file, 'r') as dataset:
        data = np.loadtxt(dataset, delimiter=',')
        x = data[:, 0]
        y = data[:, 1]

    # plt.scatter(x, y)
    # plt.show()

    model = LinearRegression(iter=10000, alpha=0.01, theta=np.zeros(2))
    model.fit(x, y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    print(intercept, slope) #excatly same as what we got from our own model


if __name__ == "__main__":
    main()