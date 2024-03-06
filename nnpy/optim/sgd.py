import nnpy
from nnpy.optim.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params: list[nnpy.Tensor], lr=1e-3, momentum=0,
                        weight_decay=0, nesterov=False) -> None:
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        for param in self.params:
            param.data -= self.defaults['lr'] * param.grad.data

    
