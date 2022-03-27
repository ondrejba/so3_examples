import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from e3nn import o3, nn, math

np.random.seed(42)
num_bases = 10

x = np.random.normal(0, 1, size=(100, 3)).astype(np.float32)
# y = x[:, 0]
y = np.sqrt(np.sum(np.square(x), axis=1))

x = torch.tensor(x)
y = torch.tensor(y)

irreps_in = o3.Irreps("1x1o")
irreps_out = o3.Irreps("1x0e")
irreps_sh = o3.Irreps.spherical_harmonics(lmax=3)
print(irreps_sh)
sh = o3.spherical_harmonics(irreps_sh, x, normalize=True, normalization="component").type(torch.float32)

tp = o3.FullyConnectedTensorProduct(
    irreps_in,
    irreps_sh,
    irreps_out,
    shared_weights=False
)
print(tp, tp.weight_numel)

length_embed = math.soft_one_hot_linspace(
    x.norm(dim=1),
    start=0.0,
    end=x.norm(dim=1).max(),
    number=num_bases,
    basis="smooth_finite",
    cutoff=True
)
length_embed = length_embed.mul(num_bases**0.5)

fc = nn.FullyConnectedNet([num_bases, 128, tp.weight_numel], torch.relu)

opt = Adam(fc.parameters(), lr=1e-3)

for i in range(10000):

    opt.zero_grad()
    w = fc(length_embed)
    pred = tp(x, sh, w)[:, 0]

    loss = torch.mean(torch.square(y - pred))
    loss.backward()
    print(loss)
    opt.step()
