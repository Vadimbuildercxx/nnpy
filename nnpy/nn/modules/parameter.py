# info https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
import nnpy

class Parameter(nnpy.Tensor):
    def __init__(self, data: nnpy.Tensor) -> None:
        assert isinstance(data, nnpy.Tensor)
        super().__init__(data.data, requires_grad=data.requires_grad)

    def parameters(self, params_in: list = None)  -> nnpy.Tensor:
        if params_in is None: params_in = []
        return params_in + [self]