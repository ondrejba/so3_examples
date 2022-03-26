import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from e3nn import o3, nn

np.random.seed(42)
x = np.random.normal(0, 1, size=(100, 3)).astype(np.float32)
y = x[:, 0]

x = torch.tensor(x)
y = torch.tensor(y)

irreps_sh = o3.Irreps.spherical_harmonics(lmax=3)
sh = o3.spherical_harmonics(irreps_sh, x, normalize=True, normalization="component").type(torch.float32)
w = torch.nn.Parameter(torch.rand(sh.shape[1]).type(torch.float32))
w.requires_grad = True

opt = Adam([w], lr=1e-2)

for i in range(1000):

    opt.zero_grad()
    pred = torch.sum(sh * w[None, :], dim=1)
    loss = torch.mean(torch.square(y - pred))
    loss.backward()
    print(loss)
    opt.step()

print(w)
