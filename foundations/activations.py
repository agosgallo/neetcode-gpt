import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array

        # Formula: 1 / (1 + e^(-z))
        return np.round(1/(1 + np.exp(-z)), 5)
    

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        res = []
        for elem in z:
            res.append(max(float(0),elem))
        return res
    #def relu(self,z):
        #return np.maximum(0,)
