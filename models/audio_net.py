import torch.nn as nn
import torch
import torch.functional as F
import numpy as np


# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=32, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len,
                      out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = x[..., :self.dim_aud].permute(1, 0).unsqueeze(0)  # 2 x subspace_dim x seq_len
        y = self.attentionConvNet(y)
        y = self.attentionNet(y.view(1, self.seq_len)).view(self.seq_len, 1)
        return torch.sum(y * x, dim=0)


# Model


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_aud=76, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(29, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size / 2)
        x = x[:, 8 - half_w:8 + half_w, :].permute(0, 2, 1)
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x).squeeze()
        return x


class DeepSpeechAudNet(nn.Module):
    def __init__(self, dim_aud=29, win_size=16):
        super(DeepSpeechAudNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud

        self.encoder_fc = nn.Sequential(
            nn.Linear(16, 1),
            nn.LeakyReLU(0.02, True),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder_fc(x).squeeze(-1)
        x = x.squeeze()
        return x
