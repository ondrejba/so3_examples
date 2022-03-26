import numpy as np
import matplotlib.pyplot as plt
import torch
from e3nn import o3, nn


def get_vector(alpha, beta):
    radius = 1.
    x = radius * np.cos(alpha) * np.sin(beta)
    y = radius * np.sin(alpha) * np.sin(beta)
    z = radius * np.cos(beta)
    return np.array([x, y, z], dtype=np.float32)


alphas = np.linspace(0., 2 * np.pi, 100)
betas = np.linspace(0., 2 * np.pi, 100)
prod = [[x, y] for x in alphas for y in betas]
vecs = torch.tensor(np.stack([get_vector(x, y) for x, y in prod], axis=0))

irreps_sh = o3.Irreps.spherical_harmonics(lmax=3)
sh = o3.spherical_harmonics(irreps_sh, vecs, normalize=True, normalization="component")

vecs = vecs.numpy()
sh = sh.numpy()

for idx in range(sh.shape[1]):

    print("Index {:d}".format(idx))

    x = sh[:, idx]
    a = np.abs(x)
    s = np.sign(x)
    cs = []
    for ss in s:
        if ss in [0, 1]:
            cs.append("r")
        else:
            cs.append("b")

    tmp_vecs = np.copy(vecs)
    tmp_vecs *= a[:, None]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(tmp_vecs[:, 0], tmp_vecs[:, 1], tmp_vecs[:, 2], c=cs)
    ax.set_zlim3d(-1.5, 1.5)
    ax.set_ylim3d(-1.5, 1.5)
    ax.set_xlim3d(-1.5, 1.5)
    plt.show()
