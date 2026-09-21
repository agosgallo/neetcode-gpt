import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        z1 = np.dot(W1,x)+ b1
        act = np.maximum(0,z1)

        z2 = np.dot(W2,act)+b2
        n = len(y_true)
        # Loss: MSE = mean((predictions - y_true)^2)
        L = 1/n * np.sum((z2-y_true)**2)
        #Backward pass
        #Derivata della loss
        dz2 = 2/n * (z2 - y_true)
        dW2 = np.outer(dz2,act)
        db2 = dz2
        dact = np.dot(dz2,W2)

        relu_mask = (z1 > 0).astype(float)
        dz1 = dact * relu_mask

        dW1 = np.outer(dz1,x)

        db1 = dz1
        
        
        #
        out = {'loss': float(np.round(L,4)),
        'dW1': np.round(dW1,4).tolist(),
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        'db1': np.round(db1,4).tolist(),
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        'dW2' : np.round(dW2,4).tolist(),
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        'db2' : np.round(db2,4).tolist()
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        }
        return out
        pass
