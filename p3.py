import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from e3nn import o3, nn, math
import layers

np.random.seed(42)
num_bases = 10

x = np.random.normal(0, 0.5, size=(100, 3)).astype(np.float32)
# y = x[:, 0]
y = np.sqrt(np.sum(np.square(x), axis=1))

x = torch.tensor(x)
y = torch.tensor(y)

irreps_in = o3.Irreps("1x1o")
irreps_out = o3.Irreps("1x0e")
irreps_sh = o3.Irreps.spherical_harmonics(lmax=3)

l1 = layers.SO3SingleVector3(irreps_in, irreps_sh, irreps_sh)
l2 = layers.SO3SingleVector3(irreps_sh, irreps_sh, irreps_sh)
l3 = layers.SO3SingleVector3(irreps_sh, irreps_sh, irreps_out)

opt = Adam(list(l1.parameters()) + list(l2.parameters()) + list(l3.parameters()), lr=1e-3)

for i in range(10000):

    opt.zero_grad()

    pred = l3(l2(l1(x)))[:, 0]

    loss = torch.mean(torch.square(y - pred))
    loss.backward()
    print(loss)
    opt.step()
