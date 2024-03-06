import numpy as np

# info https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

class Tensor:
    def __init__(self, data, _children=(), _op="", label = "", requires_grad= False) -> None:

        self.data: np.ndarray = np.array(object=data)
        if _op != "grad":
            self.grad: Tensor = Tensor(np.zeros(shape=self.data.shape), _op="grad")

        self.requires_grad: bool = requires_grad

        self._backward = lambda: None

        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    
    ### Fix bug with shapes (1, )  and () 
    def __add__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op="+", requires_grad=self.requires_grad)

        if self.requires_grad:
            def _add_backward():

                self_shape_extended = np.pad(np.array(self.shape), (out.ndim - self.ndim, 0))

                diff_self_out = np.minimum(np.abs(np.array(object=out.shape) - self_shape_extended), 1)
                
                ## Todo need to fix ones mul
                _grad_self = np.ones(shape=out.shape) * out.grad.data
                for i, x in enumerate(np.nditer(diff_self_out)):
                    if x.item(0) != 0:
                        _grad_self = _grad_self.sum(i, keepdims=True)

                self.grad.data += _grad_self.reshape(self.grad.shape) # * out.grad

                other_shape_extended = np.pad(np.array(other.shape), (out.ndim - other.ndim, 0))

                diff_other_out = np.minimum(np.abs(np.array(object=out.shape) - other_shape_extended), 1)
                
                ## Todo need to fix ones mul
                _grad_other = np.ones(shape=out.shape) * out.grad.data
                for i, x in enumerate(np.nditer(diff_other_out)):
                    if x.item(0) != 0:
                        _grad_other = _grad_other.sum(i, keepdims=True)

                other.grad.data += _grad_other.reshape(other.grad.shape) #+ 1.0 * out.grad

            out._backward  = _add_backward

        return out
    
    def __radd__(self, other: "Tensor") -> "Tensor": 
        return self + other
    
    ### Fix broadcast shape issue with shapes like [2], [2,2]
    def __mul__(self, other: "Tensor") -> "Tensor":
        #other = [other] if isinstance(other, (int, float)) else other
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,(self, other), _op="*", requires_grad=self.requires_grad)

        if self.requires_grad:
            def _mul_backward():
                diff_self_out = np.minimum(
                    np.abs(np.array(self.shape) - np.array(object=out.shape))
                    , 1)

                _grad_self = np.broadcast_to(array = other.data * out.grad.data, shape=out.shape)
                for i, x in enumerate(np.nditer(diff_self_out)):
                    if x.item(0) != 0:
                        _grad_self = _grad_self.sum(i, keepdims=True)

                self.grad.data += _grad_self.reshape(self.grad.shape)
            
                diff_other_out = np.minimum(
                    np.abs(np.array(other.shape) - np.array(object=out.shape))
                    , 1)

                _grad_other = np.broadcast_to(array = self.data * out.grad.data, shape=out.shape)
                for i, x in enumerate(np.nditer(diff_other_out)):
                    if x.item(0) != 0:
                        _grad_other = _grad_other.sum(i, keepdims=True)

                other.grad.data += _grad_other.reshape(other.grad.shape)

            out._backward  = _mul_backward

        return out
    
    def __pow__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.power(self.data, other.data),(self, ), _op=f"**{other}", requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _pow_backward():
                # self.grad += other.data * (np.power(self.data, other.data - 1))  * out.grad
                # other.grad += np.power(self.data, other.data) * np.log(self.data) * out.grad

                self_shape_extended = np.pad(np.array(self.shape), (out.ndim - self.ndim, 0))

                diff_self_out = np.minimum(np.abs(np.array(object=out.shape) - self_shape_extended), 1)

                _grad_self = np.broadcast_to(
                    array = other.data * (np.power(self.data, other.data - 1))  * out.grad.data,
                    shape=out.shape
                    )
                
                for i, x in enumerate(np.nditer(diff_self_out)):
                    if x.item(0) != 0:
                        _grad_self = _grad_self.sum(i, keepdims=True)

                self.grad.data += _grad_self.reshape(self.grad.shape)


                other_shape_extended = np.pad(np.array(other.shape), (out.ndim - other.ndim, 0))

                diff_other_out = np.minimum(np.abs(np.array(object=out.shape) - other_shape_extended), 1)
                
                _grad_other = np.broadcast_to(
                    array = np.power(self.data, other.data) * np.log(self.data) * out.grad.data, 
                    shape=out.shape
                    )
                
                for i, x in enumerate(np.nditer(diff_other_out)):
                    if x.item(0) != 0:
                        _grad_other = _grad_other.sum(i, keepdims=True)

                other.grad.data += _grad_other.reshape(other.grad.shape)

            out._backward  = _pow_backward

        return out
    
    def __rmul__(self, other: "Tensor") -> "Tensor":
        return self * other
    
    def __truediv__(self, other: "Tensor") -> "Tensor":
        return self * other** (-np.ones(shape=other.data.shape))
    
    def __neg__(self) -> "Tensor":
        return self * (-np.ones(shape=self.data.shape))
    
    def __sub__(self, other: "Tensor") -> "Tensor":
        return self + (-other)
    
    def __matmul__(self, other: "Tensor") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)

        product = np.matmul(self.data, other.data)
        out = Tensor(data=product, _children=(self, other), _op="@", requires_grad=self.requires_grad)

        if self.requires_grad:
            def _mm_backward():
                self.grad.data += out.grad.data @ other.data.T 
                other.grad.data +=  self.data.T @ out.grad.data 
            out._backward  = _mm_backward

        return out
    
    @property
    def T(self):
        out = Tensor(data=self.data.T, _children=(self, ), _op="T", requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _transpose_backward():
                self.grad.data += out.grad.data.T
            out._backward  = _transpose_backward

        return out
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    def sum(self, dim = None) -> "Tensor":
        out = Tensor(data = np.squeeze(np.sum(self.data, axis=dim, keepdims=True)), _children=(self,), _op = "sum", requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _sum_backward():
                self.grad.data += 1.0 * out.grad.data
            out._backward = _sum_backward

        return out
    
    # Need check
    def tanh(self) -> "Tensor":
        x = self.data
        t = (np.exp(2*x) - 1)/(np.exp(2*x) + 1)
        out = Tensor(t, (self,), "tanh", requires_grad=self.requires_grad)

        if self.requires_grad:
            def _tanh_backward():
                self.grad.data += (1 - t ** 2) * out.grad.data

            out._backward  = _tanh_backward
        
        return out
    
    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), (self, ), "exp", requires_grad=self.requires_grad)

        if self.requires_grad:
            def _exp_backward():
                self.grad.data += out.data * out.grad.data
            
            out._backward = _exp_backward

        return out
    
    def zero_grad(self):
        self.grad.data = np.zeros(self.data.shape)
    
    def backward(self, external_grad: "Tensor" = None) -> None:
        #external_grad = np.array(external_grad)
        if external_grad is not None:
            assert isinstance(external_grad, Tensor)

        topo: list[Tensor] = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        if external_grad is None:
            self.grad = tensor([1.0], requires_grad=True)
        else:
            self.grad = external_grad

        if self.data.shape != ( ) and self.data.shape != self.grad.shape:
            raise Exception("Wrong shape of gradient expected scalar value, please define ``external_grad`` ")

        for node in reversed(topo):
            node._backward()

def tensor(data, requires_grad = False) -> Tensor:
    return Tensor(data=data, requires_grad = requires_grad)

def ones(size, requires_grad = False) -> Tensor:
    return Tensor(data=np.ones(shape=size), requires_grad = requires_grad)

def zeros(size, requires_grad = False) -> Tensor:
    return Tensor(data=np.zeros(shape=size), requires_grad = requires_grad)

def rand(*size, requires_grad = False) -> Tensor:
    return Tensor(data=np.random.rand(*size), requires_grad = requires_grad)

def maximum(input: Tensor, other: Tensor) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(np.maximum(input.data, other.data),(input, ), _op="maximum", requires_grad=input.requires_grad)
    
    grad = np.where(out.data > 0., 1, out.data)

    if input.requires_grad:
        def _backward():
            input.grad.data += out.grad.data * grad
        out._backward  = _backward

    return out

def sum(input: Tensor, dim=None, keepdim=False) -> Tensor:
    out = Tensor(data=np.sum(input.data, axis=dim, keepdims=keepdim), _children=(input,), _op = "sum", requires_grad=input.requires_grad)

    if input.requires_grad:
        def _sum_backward():
            input.grad.data += 1.0 * out.grad.data
        out._backward  = _sum_backward

    return out

def log(input: Tensor) -> Tensor:
    out = Tensor(data=np.log(input.data), _children=(input,), _op = "log", requires_grad=input.requires_grad)
        
    if input.requires_grad:
        def _log_backward():
            input.grad.data += (1 / out.data)* out.grad.data
        out._backward  = _log_backward

    return out

def mean(input: Tensor, dim=None) -> Tensor:
    out = Tensor(data=np.mean(input.data, axis=dim), _children=(input,), _op = "mean", requires_grad=input.requires_grad)
        
    if input.requires_grad:
        def _mean_backward():
            input.grad.data += 1.0 * out.grad.data / input.data.size * out.data.size
        out._backward  = _mean_backward

    return out


def square(input: Tensor) -> Tensor:
    out = Tensor(data=np.square(input.data), _children=(input,), _op = "square", requires_grad=input.requires_grad)
        
    if input.requires_grad:
        def _square_backward():
            input.grad.data += 1.0 * out.grad.data * 2 * input.data 
        out._backward  = _square_backward

    return out