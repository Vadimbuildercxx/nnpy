import numpy as np

class CrossEntropyLoss():
    """
    This criterion computes the cross entropy loss between input logits and target.
    Args:
        weight (optional): a manual rescaling weight given to each class.
        reduction (optional): Different types of reduction ``sum``, ``mean``
    """
    def __init__(self, weight: np.float_ = 1, reduction = "mean") -> None:
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: np.ndarray, target: np.ndarray) -> np.float_:
        loss = -self.weight * np.log(input) * target
        if self.reduction == "mean":
            loss = np.mean(loss)
        elif self.reduction == "sum":
            loss = np.sum(loss)
        else:
            raise Exception("Wrong reduction type")
        return loss