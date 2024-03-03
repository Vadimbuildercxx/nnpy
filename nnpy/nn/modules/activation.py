import nnpy
from nnpy.nn.modules.module import Module

class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: nnpy.Tensor) -> nnpy.Tensor:
        return nnpy.maximum(tensor, nnpy.tensor([0]))
    
    def __call__(self, tensor: nnpy.Tensor) -> nnpy.Tensor:
        return self.forward(tensor)

class Softmax(Module):
    def __init__(self, dim = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor: nnpy.Tensor) -> nnpy.Tensor:
        exp = tensor.exp()
        return exp / nnpy.sum(exp, dim = self.dim, keepdim=True)
    
    def __call__(self, tensor: nnpy.Tensor) -> nnpy.Tensor:
        return self.forward(tensor)