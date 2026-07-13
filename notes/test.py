from tinygrad.tensor import Tensor

t = Tensor.ones(4) + 2
t_fin = t.tolist()
print(t_fin)
