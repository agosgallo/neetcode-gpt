import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        zio_massimo = max(z)
        out = np.exp(z - zio_massimo) 
        den = sum(np.exp(z - zio_massimo))
        return np.round(out/den,4)
        # return np.round(your_answer, 4)
        pass
