import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1,256,16))
    o = cb(x)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        modules = []

        modules.append(
            nn.Sequential(nn.Conv1d(10, out_channels=6, kernel_size=5, padding=2),nn.GroupNorm(2, 6),nn.Conv1d(6, out_channels=4, kernel_size=5, padding=2),nn.GroupNorm(2, 4),nn.Conv1d(4, out_channels=2, kernel_size=5, padding=2),nn.GroupNorm(2, 2),nn.Conv1d(2, out_channels=1, kernel_size=5, padding=2),nn.GroupNorm(1, 1),nn.Flatten(1, -1),nn.Linear(16, 1),nn.Sigmoid(),)
        )
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)

class MLPConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels=10, 
            out_channels=1, 
            cond_dim=256,
            kernel_size=5,
            n_groups=2,
            cond_predict_scale=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, 6, kernel_size, n_groups),
            Conv1dBlock(6, out_channels, kernel_size, n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
        self.mlp_layers= nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        print('COND SHAPE', cond.shape)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        out = self.mlp_layers(out)
        return out