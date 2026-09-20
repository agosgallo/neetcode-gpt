import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        z = float(np.dot(x,w) + b)
        # Pre-activation: z = dot(x, w) + b
        if activation == 'sigmoid':

        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
            res = 1 / (1 + np.exp(-z))

        elif activation == 'relu':
            
        # ReLU: max(0, z)
            res = np.maximum(0,z)
        
        return round(res, 5)
        pass
