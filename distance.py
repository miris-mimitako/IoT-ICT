import numpy as np
import pandas as pd
from sklearn import preprocessing

class Distance:
    def __init__(self) -> None:
        pass

    def euclidean(self, x1, x2):
        """
        # Euclidean distance
        x1, x2: numpy array
        return: float
        """
        if x1.shape != x2.shape:
            raise ValueError("x1 and x2 must have the same shape")

        return np.sqrt(np.sum((x1 - x2)**2))
    
    def mean_sigma_column(self, arr):
        """
        # Mean and sigma of each column
        arr: numpy array
        return: numpy array
        """
        return np.mean(arr, axis=0), np.std(arr, axis=0)
    
    def standardized_euclidean(self, arr):
        """
        # Standardized Euclidean distance
        arr: numpy array
        return: numpy array
        """
        arr_standardized =  preprocessing.scale(arr, axis=0)
        print(arr_standardized.std(axis=0))
        return np.sqrt(np.sum(arr_standardized**2, axis=1))

    def manhattan(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def minkowski(self, x1, x2, p):
        return np.sum(np.abs(x1 - x2)**p)**(1/p)

    def cosine(self, x1, x2):
        return np.dot(x1, x2) / (np.sqrt(np.sum(x1**2)) * np.sqrt(np.sum(x2**2)))

    def jaccard(self, x1, x2):
        return np.sum(np.minimum(x1, x2)) / np.sum(np.maximum(x1, x2))

    def hamming(self, x1, x2):
        return np.sum(x1 != x2) / len(x1)

    def correlation(self, x1, x2):
        return np.sum((x1 - x1.mean()) * (x2 - x2.mean())) / (np.sqrt(np.sum((x1 - x1.mean())**2)) * np.sqrt(np.sum((x2 - x2.mean())**2)))

    def mahalanobis(self, x1, x2, cov):
        return np.sqrt(np.dot(np.dot((x1 - x2).T, cov), (x1 - x2)))

    def maha(self, x1, x2, cov):
        return np.sqrt(np.dot(np.dot((x1 - x2).T, np.linalg.inv(cov)), (x1 - x2)))

    def maha2(self, x1, x2, cov):
        return np.sqrt(np.dot(np.dot((x1 - x2).T, np.linalg.pinv(cov)), (x1 - x2)))

    def maha3(self, x1, x2, cov):
        return np.sqrt(np.dot(np.dot((x1 - x2).T, np.linalg.pinv(cov + 0.1 * np.eye(cov.shape[0]))), (x1 - x2)))

if __name__ == "__main__":
    D = Distance()
    x1 = np.array([[10, 2, 3],[1, 3, 3],[4, 5, 6],[4, 7, 8]])
    x2 = np.array([[4, 5, 6],[4, 5, 6]])
    cov = np.cov(x1)
    print(D.mean_sigma_column(x1))