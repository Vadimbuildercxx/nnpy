import nnpy

class Optimizer:
    def __init__(self, params: list[nnpy.Tensor], defaults: dict) -> None:
        self.params = params
        self.defaults = defaults

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
