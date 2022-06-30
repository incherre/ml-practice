'''A tool to generate some test data for binary classification.'''
import numpy as np
import os
import csv

class Distribution:
    def __init__(self, label, mean, covariance):
        self.label = label
        self.mean = mean
        self.covariance = covariance

    def sample(self):
        return np.hstack((np.random.multivariate_normal(self.mean, self.covariance), self.label))

if __name__ == '__main__':
    distributions = [
        Distribution([0], [-1, 1], [[1, 0], [0, 10]]),
        Distribution([1], [1, -1], [[1, 0], [0, 10]]),
    ]
    distribution_weights = [
        0.5,
        0.5,
    ]

    train_samples = 900
    train_path = os.path.abspath(os.path.join(".", "data", "train.csv"))

    test_samples = 100
    test_path = os.path.abspath(os.path.join(".", "data", "test.csv"))

    train_file = open(train_path, 'w', newline='', encoding='utf-8')
    train_csv = csv.writer(train_file)
    for i in range(train_samples):
        record = np.random.choice(distributions, p=distribution_weights).sample()
        train_csv.writerow(record)
    train_file.close()

    test_file = open(test_path, 'w', newline='', encoding='utf-8')
    test_csv = csv.writer(test_file)
    for i in range(test_samples):
        record = np.random.choice(distributions, p=distribution_weights).sample()
        test_csv.writerow(record)
    test_file.close()
