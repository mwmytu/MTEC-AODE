import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj):
        super(ODEFunc, self).__init__()
        self.adj = adj
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))

        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)

    def forward(self, t, x):
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        xa = torch.einsum('ij, kjln->kiln', self.adj, x)

        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.einsum('ijkl, mn->ijkm', x, w)
        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        xw2 = torch.einsum('ijkl, kn->ijnl', x, w2)

        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        return f


class ODEblock(nn.Module):
    def __init__(self, odefunc, t_scales):
        super(ODEblock, self).__init__()
        self.t_scales = torch.tensor(t_scales)
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x_aug):
        for i in range(len(self.t_scales) - 1):
            t = self.t_scales[i:i + 2].type_as(x_aug)
            z = odeint(self.odefunc, x_aug, t, method='euler')[1]
        return z

class ODEG(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj, time_scales):
        super(ODEG, self).__init__()
        self.num_zeros = 10
        self.odeblock = ODEblock(ODEFunc(feature_dim + 10, temporal_dim, adj), t_scales=time_scales)
        self.feature_dim_before = None
        self.feature_dim_after = None

    def forward(self, x):
        batch_size, num_nodes, num_timesteps, num_features = x.shape
        if(num_features == 64):
            aug = torch.zeros(batch_size, num_nodes, num_timesteps, self.num_zeros).to(x.device)
            x_aug = torch.cat([x, aug], 3)
            self.feature_dim_before = x_aug
        else:
            x_aug = x
        self.odeblock.set_x0(x_aug)
        z = self.odeblock(x_aug)
        self.feature_dim_after = z
        return F.relu(z)



