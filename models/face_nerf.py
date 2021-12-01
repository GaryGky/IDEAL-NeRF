import einops
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class FaceNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_views=27, dim_aud=64, dim_latent=0, dim_expr=0,
                 output_ch=4, skips=None, use_viewdirs=True):
        """
        """
        super(FaceNeRF, self).__init__()
        if skips is None:
            skips = [4]
        self.D = D
        self.W = W
        self.input_xyz_ch = input_ch
        self.input_views_ch = input_ch_views
        self.dim_aud = dim_aud
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.dim_expr = dim_expr
        self.dim_latent = dim_latent

        # 这里直接把图像通道数和音频特征拼接在一起输入网络
        input_ch_all = self.input_xyz_ch + self.dim_aud + self.dim_expr + self.dim_latent
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch_all, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch_all, W) for i in range(D - 1)])

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W + self.dim_expr, W // 2)] + [nn.Linear(W // 2, W // 2) for i in range(D // 4)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, aud, expr=None, latent_code=None):
        input_pts, input_views = torch.split(x, [self.input_xyz_ch, self.input_views_ch], dim=-1)
        initial = input_pts

        if aud is not None:
            aud = einops.repeat(aud, 'c -> b c', b=x.shape[0])
            initial = torch.cat([initial, aud], dim=-1)

        if expr is not None:
            expr = expr * 1 / 3
            expr = einops.repeat(expr, 'c -> b c', b=x.shape[0])
            initial = torch.cat([initial, expr], dim=-1)

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
            h = torch.cat([feature, input_views], -1)
            if expr is not None:
                h = torch.cat([h, expr], dim=-1)

            for i, layer in enumerate(self.views_linears):
                h = layer(h)
                h = F.relu(h, inplace=True)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


if __name__ == '__main__':
    face_nerf = FaceNeRF(D=8, W=256, input_ch=63, dim_aud=0, output_ch=256, skips=None,
                         dim_latent=32, dim_expr=79, input_ch_views=27, use_viewdirs=True)
    print(face_nerf)