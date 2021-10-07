import numpy as np
from util import feature_normalizer


def compute_theta(x, y):
    x_new = np.column_stack((np.ones(x.shape[0]), x))
    return np.dot(np.linalg.inv(np.dot(x_new.T,  x_new)), (np.dot(np.transpose(x_new), y)))

def predict(theta, x):
    x_new = np.column_stack((np.ones(x.shape[0]), x))
    return np.dot(x_new, theta)

def main():
    path_file = '..\machine-learning\LinearRegression\ex1data2.txt'

    with open(path_file, 'r') as dataset:
        data = np.loadtxt(dataset, delimiter=',')
        x = data[:, :-1]
        y = data[:, -1]

    x = feature_normalizer(x)

    theta = compute_theta(x, y)
    print(theta)

if __name__ == '__main__':
    main()