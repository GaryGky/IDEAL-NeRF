import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_ch, attn_output_ch=256):
        super(SelfAttention, self).__init__()
        self.input_ch = input_ch
        self.attn_output_ch = attn_output_ch

        self.to_k = nn.Linear(in_features=self.input_ch, out_features=self.attn_output_ch)
        self.to_q = nn.Linear(in_features=self.input_ch, out_features=self.attn_output_ch)
        self.to_v = nn.Linear(in_features=self.input_ch, out_features=self.attn_output_ch)
        self.scale = 1 / math.sqrt(self.attn_output_ch)

        self.to_x = nn.Linear(in_features=attn_output_ch, out_features=input_ch)

    def forward(self, x):
        Q = self.to_q(x)
        K = self.to_k(x)
        V = self.to_v(x)

        attn = nn.Softmax(dim=-1)(torch.mm(Q, K.permute(1, 0))) * self.scale
        attn_out = torch.mm(attn, V)
        attn_out = self.to_x(attn_out) + x

        return attn_out


if __name__ == '__main__':
    sample = torch.randn(1024, 208)
    self_attention = SelfAttention(input_ch=208, attn_output_ch=256)
    out = self_attention(sample)
    print(out.shape)
    print(out)
