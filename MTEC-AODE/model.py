import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from odegcn import ODEG

class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # permute shape to (B, F, N, T)
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y

class GCN(nn.Module):
    def __init__(self, A_hat, in_channels, out_channels, ):
        super(GCN, self).__init__()
        self.A_hat = A_hat
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset()
    def reset(self):
        stdv = 1. / math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-stdv, stdv)
    def forward(self, X):
        y = torch.einsum('ij, kjlm-> kilm', self.A_hat, X)
        return F.relu(torch.einsum('kjlm, mn->kjln', y, self.theta))

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, A_hat):
        super(STGCNBlock, self).__init__()
        self.A_hat = A_hat
        self.temporal1 = TemporalConvNet(num_inputs=in_channels,
                                         num_channels=out_channels)
        self.odeg = ODEG(out_channels[-1], 12, A_hat, time_scales = [6,8])
        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1],
                                         num_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X):
        t = self.temporal1(X)
        t = self.odeg(t)
        t = self.temporal2(F.relu(t))
        return self.batch_norm(t)


class ODEGCN(nn.Module):

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, A_sp_hat, A_se_hat):
        super(ODEGCN, self).__init__()
        self.sp_blocks = nn.ModuleList(
            [nn.Sequential(
                STGCNBlock(in_channels=num_features, out_channels=[64, 32, 64],
                           num_nodes=num_nodes, A_hat=A_sp_hat),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                           num_nodes=num_nodes, A_hat=A_sp_hat)) for _ in range(3)
            ])
        self.se_blocks = nn.ModuleList([nn.Sequential(
            STGCNBlock(in_channels=num_features, out_channels=[64, 32, 64],
                       num_nodes=num_nodes, A_hat=A_se_hat),
            STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                       num_nodes=num_nodes, A_hat=A_se_hat)) for _ in range(3)
        ])

        self.pred = nn.Sequential(
            nn.Linear(num_timesteps_input * 64, num_timesteps_output * 32),
            nn.ReLU(),
            nn.Linear(num_timesteps_output * 32, num_timesteps_output)
        )

    def forward(self, x):

        outs = []
        for blk in self.sp_blocks:
            outs.append(blk(x))
        for blk in self.se_blocks:
            outs.append(blk(x))
        outs = torch.stack(outs)
        x = torch.max(outs, dim=0)[0]
        x = x[..., :64]
        x = x.reshape((x.shape[0], x.shape[1], -1))
        return self.pred(x)
