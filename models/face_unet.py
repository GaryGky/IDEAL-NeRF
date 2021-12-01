import queue

import torch.nn as nn
import torch
import torch.nn.functional as F

"""
use_global
use_render_pose
含义都不明确的情况下，先不加入
"""


class FaceUNetCNN(nn.Module):
    def __init__(self, embed_ln, H=450, W=450, input_ch=66, use_global=False, use_render_pose=False):
        super(FaceUNetCNN, self).__init__()
        self.embed_ln = embed_ln
        self.use_global = use_global
        self.use_render_pose = use_render_pose

        # image information
        self.H = H
        self.W = W
        self.input_ch = input_ch

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_ch, 64, kernel_size=7, stride=2, padding=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, dilation=1, padding=1, output_padding=1),
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, dilation=2, padding=2, output_padding=1),
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, dilation=4, padding=4, output_padding=1),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, dilation=8, padding=8, output_padding=1),
        )

    def forward(self, x):
        embeded_rgb = x[..., :self.embed_ln]
        x = torch.transpose(x, 1, 3)
        embeded_rgb = torch.transpose(embeded_rgb, 1, 3)
        stack = []
        for layer in self.encoder:
            x = F.relu(layer(x), inplace=True)
            stack.append(x)

        _ = stack.pop()

        for layer in self.decoder:
            x = F.relu(layer(x), inplace=True)
            if len(stack) > 0:
                x_prev = stack.pop()
                x = torch.cat([x, x_prev], 1)

        x = F.pad(x, pad=(1, 1, 1, 1), mode="replicate")
        return torch.cat([x, embeded_rgb], 1)


if __name__ == '__main__':
    sample = torch.ones((1, 450, 450, 3))

    print(sample.shape)

    face_unet = FaceUNetCNN(33, 450, 450, 3)
    output = face_unet(sample)
    print(face_unet.encoder.modules())
    print(face_unet.decoder.modules())

    print(output.shape)
