import nnpy
from nnpy.nn.modules.module import Module
from nnpy.nn.modules.parameter import Parameter

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True) -> None:
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        if self.bias:
            self.b = Parameter(nnpy.zeros(out_features, requires_grad=True))
        self.weight = Parameter(nnpy.rand(out_features, in_features, requires_grad=True))


    def forward(self, tensor: nnpy.Tensor) -> nnpy.Tensor:
        out: nnpy.Tensor = tensor @ self.weight.T
        if self.bias:
            out = out + self.b
        return out
    
    def __call__(self, tensor: nnpy.Tensor) -> nnpy.Tensor:
        return self.forward(tensor)
