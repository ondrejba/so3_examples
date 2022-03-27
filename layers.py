import torch
from torch import nn
from e3nn import o3, math
import e3nn.nn as e3nnnn


class SO3SingleVector3(nn.Module):

    NUM_BASES = 10
    NUM_HIDDEN_NEURONS = 256

    def __init__(self, irreps_in, irreps_sh, irreps_out):

        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_sh = irreps_sh
        self.irreps_out = irreps_out

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in,
            irreps_sh,
            irreps_out,
            shared_weights=False
        )

        # TODO: why is the hidden dim length_embed.shape[1]
        self.fc = e3nnnn.FullyConnectedNet(
            [self.NUM_BASES, self.NUM_HIDDEN_NEURONS, self.tp.weight_numel],
            torch.relu
        )

    def forward(self, x):

        x_norm = x.norm(dim=1)

        length_embed = math.soft_one_hot_linspace(
            x_norm,
            start=0.0,
            end=3.0,
            number=self.NUM_BASES,
            basis="smooth_finite",
            cutoff=True
        )
        length_embed = length_embed.mul(self.NUM_BASES ** 0.5)

        sh = o3.spherical_harmonics(
            self.irreps_sh,
            x,
            normalize=True,
            normalization="component"
        ).type(torch.float32)

        weights = self.fc(length_embed)
        return self.tp(x, sh, weights)
