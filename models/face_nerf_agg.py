import einops
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class FaceNeRFAgg(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_views=27, dim_agg=64, dim_aud=64, dim_expr=0,
                 dim_latent=0, output_ch=4, skips=None, use_viewdirs=True):
        """
        """
        super(FaceNeRFAgg, self).__init__()
        if skips is None:
            skips = [4]
        self.D = D
        self.W = W
        self.input_xyz_ch = input_ch
        self.input_views_ch = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.dim_latent = dim_latent
        self.dim_agg = dim_agg
        self.dim_aud = dim_aud
        self.dim_expr = dim_expr

        # 聚合aud和expr
        self.agg_linears = nn.ModuleList(
            [nn.Linear(self.dim_expr + self.dim_aud, self.dim_agg),
             nn.Linear(self.dim_agg, self.dim_agg)])

        # 把融合的特征直接拼进去
        input_ch_all = self.input_xyz_ch + self.dim_agg + self.dim_latent
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch_all, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch_all, W)
                                            for i in range(D - 1)])

        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W + self.dim_agg, W // 2)] +
            [nn.Linear(W // 2, W // 2) for i in range(D // 4)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, aud, expr=None, latent_code=None):
        input_pts, input_views = torch.split(x, [self.input_xyz_ch, self.input_views_ch], dim=-1)

        # 融合aud和expr
        aud = einops.repeat(aud, 'c -> b c', b=x.shape[0])
        if expr is not None:
            expr = expr * 1 / 3
            expr = einops.repeat(expr, 'c -> b c', b=x.shape[0])
        agg_input = torch.cat([aud, expr], dim=-1)
        h = agg_input
        for layer in self.agg_linears:
            h = layer(h)

        agg_feat = h

        initial = torch.cat([input_pts, agg_feat], dim=-1)
        if latent_code is not None:
            latent_code = einops.repeat(latent_code, 'c -> b c', b=x.shape[0])
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
            h = torch.cat([feature, input_views, agg_feat], -1)

            for i, layer in enumerate(self.views_linears):
                h = layer(h)
                h = F.relu(h, inplace=True)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs