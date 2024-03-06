import nnpy

class CrossEntropyLoss():
    """
    This criterion computes the cross entropy loss between input logits and target.
    Args:
        weight (optional): a manual rescaling weight given to each class.
        reduction (optional): Different types of reduction ``sum``, ``mean``
    """
    def __init__(self, weight = 1, reduction = "mean") -> None:
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: nnpy.Tensor, target: nnpy.Tensor) -> nnpy.Tensor:
        loss = -self.weight * nnpy.log(input) * target
        if self.reduction == "mean":
            loss = nnpy.mean(loss)
        elif self.reduction == "sum":
            loss = nnpy.sum(loss)
        else:
            raise Exception("Wrong reduction type")
        return loss
    
    def __call__(self, input: nnpy.Tensor, target: nnpy.Tensor) -> nnpy.Tensor:
        return self.forward(input, target)