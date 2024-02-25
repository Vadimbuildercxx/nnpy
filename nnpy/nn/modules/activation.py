import nnpy

class ReLU():
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: nnpy.Tensor) -> nnpy.Tensor:
        return nnpy.maximum(tensor, nnpy.tensor([0]))
    
    def __call__(self, tensor: nnpy.Tensor) -> nnpy.Tensor:
        return self.forward(tensor)
