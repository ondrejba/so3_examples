import numpy as np
import matplotlib.pyplot as plt
import torch
from e3nn import o3, nn, math


num_bases = 10
max_radius = 2.
x = torch.linspace(0., 2., 1000)
y = math.soft_one_hot_linspace(
    x,
    start=0.,
    end=max_radius,
    number=num_bases,
    basis="smooth_finite",
    cutoff=True
)
print(y.shape)
plt.plot(x, y)
plt.show()
