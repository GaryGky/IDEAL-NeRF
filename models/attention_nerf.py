import einops
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class AttentionNeRF(nn.Module):
    def __init__(self, D=8, W=256, output_ch=4, skips=None, use_viewdirs=True,
                 input_ch=256, input_ch_views=27, input_ch_xyz=63, dim_latent=0, ):
        """
        """
        super(AttentionNeRF, self).__init__()
        if skips is None:
            skips = [4]
        self.D = D
        self.W = W
        self.input_attn_ch = input_ch
        self.input_views_ch = input_ch_views
        self.input_ch_xyz = input_ch_xyz
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.dim_latent = dim_latent

        input_ch_all = self.input_attn_ch + self.dim_latent + self.input_ch_xyz
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch_all, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch_all, W)
                                            for i in range(D - 1)])

        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W // 2)] +
            [nn.Linear(W // 2, W // 2) for i in range(D // 4)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, latent_code=None):
        input_pts, input_views = torch.split(x, [self.input_attn_ch + self.input_ch_xyz, self.input_views_ch], dim=-1)
        initial = input_pts
        if latent_code is not None:
            latent_code = einops.repeat(latent_code, 'c ->h c', h=x.shape[0])
            initial = torch.cat([initial, latent_code], dim=-1)

        h = initial
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.relu(h, inplace=True)
            if i in self.skips:
                h = torch.cat([initial, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = h
            h = torch.cat([feature, input_views], -1)

            for i, layer in enumerate(self.views_linears):
                h = layer(h)
                h = F.relu(h, inplace=True)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs