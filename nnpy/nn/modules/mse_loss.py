import nnpy

class MSELoss():
    """
    This criterion computes the mean squared loss between input logits and target.
    Args:
        weight (optional): a manual rescaling weight given to each class.
        reduction (optional): Different types of reduction ``sum``, ``mean``
    """
    def __init__(self) -> None:
        pass

    def forward(self, input: nnpy.Tensor, target: nnpy.Tensor) -> nnpy.Tensor:
        loss = nnpy.mean(nnpy.square(input - target))
        return loss
    
    def __call__(self, input: nnpy.Tensor, target: nnpy.Tensor) -> nnpy.Tensor:
        return self.forward(input, target)