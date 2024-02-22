import nnpy 

# class Module:
#     def __init__(self) -> None:
#         pass


# class Linear(Module):
#     def __init__(self, in_features: int, out_features: int, bias=True) -> None:
#         super().__init__()

#         self.in_features = in_features
#         self.out_features = out_features
#         self.bias = bias

#         if self.bias:
#             self.b = nnpy.zeros(out_features)

#         self.w = nnpy.rand(in_features, out_features)


data_a = [
    [3., 4.],
    [3., 4.],
    [3., 4.]
    ]
data_b = [
    [4., 5., 5.],
    [4., 5., 5.]
    ]

external_grad = [
    [1.0, 1.0],
    [1.0, 1.0]]

a = nnpy.tensor(data_a, requires_grad=True)
b = nnpy.tensor(data_b, requires_grad=True)

q = a.T @ b.T
print(q)
q.backward(external_grad)


print(a.grad)
print(b.grad)


