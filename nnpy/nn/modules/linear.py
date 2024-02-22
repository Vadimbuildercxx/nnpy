import numpy as np

class Linear():
    def __init__(self, in_features: int, out_features: int, bias: bool=True) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        if bias:
            self.b = np.zeros(shape=out_features)
        self.weights = np.random.rand(in_features, out_features)

    def forward(self, array: np.ndarray) -> np.ndarray:
        out = np.matmul(array, self.weights)
        if self.bias:
            out += self.b
        return out
    
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return self.forward(array)
