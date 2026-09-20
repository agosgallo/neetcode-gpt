import numpy as np
from numpy.typing import NDArray
from numpy import log

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        n = len(y_true)
        eps = 1e-7
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        y_pred= np.clip(y_pred, eps , 1-eps)
        res = - 1/n*np.sum(y_true*log(y_pred)+(1-y_true)*log(1-y_pred))
        return np.round(res,4)
        # return round(your_answer, 4)
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        n = len(y_true)
        eps = 1e-7
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        y_pred = np.clip(y_pred,eps,1-eps)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        internal = np.sum(y_true*log(y_pred))
        res =  -1/n * np.sum(internal)
        return round(res, 4)
        pass
