#info https://discuss.pytorch.org/t/how-does-parameter-work/11960/4
#info https://pytorch.org/docs/0.3.0/nn.html#torch.nn.ModuleList
from typing import Any
from nnpy.nn.modules.parameter import Parameter

class Module:
    def __init__(self) -> None:
        pass

    def parameters(self, params_in: list = None) -> None:
        if params_in is None: params_in = []
        class_variables = [attribute for attribute in vars(self)
                   if isinstance(getattr(self, attribute), (Parameter, Module))]
        
        params = []
        for attr in class_variables:
            l = getattr(self, attr).parameters(params_in)
        
            params.extend(l)

        return params_in + params
    
    def forward(self, *args):
        raise NotImplementedError()
    
    def __call__(self, *args: Any):
        return self.forward(*args)
