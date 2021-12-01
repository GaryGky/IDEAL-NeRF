"""注意力机制模块"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class AttentionSets(nn.Module):
    def __init__(self, W=256, input_ch=3, attention_output_length=512):
        super(AttentionSets, self).__init__()

        self.W = W
        self.input_ch = input_ch
        self.attention_output_length = attention_output_length

        self.dense_layer_1 = nn.Sequential(
            nn.Linear(in_features=self.input_ch, out_features=self.W),  # dense1
            nn.Linear(in_features=self.W, out_features=self.W),  # dense2
            nn.Linear(in_features=self.W, out_features=self.W),  # dense3
        )

        self.dense_layer_2 = nn.Linear(in_features=self.W + self.input_ch,
                                       out_features=self.attention_output_length)  # dense4

        self.dense_layer_3 = nn.Linear(in_features=self.attention_output_length,
                                       out_features=self.attention_output_length)  # dense5

        self.dense_layer_4 = nn.Linear(in_features=1,
                                       out_features=self.attention_output_length)  # dense6

    def forward(self, inputs):
        x = inputs
        for layer in self.dense_layer_1:
            x = layer(x)
            x = F.relu(x, inplace=True)

        x = torch.cat([x, inputs], -1)
        x = self.dense_layer_2(x)
        x = F.leaky_relu(x, inplace=True)

        mask = self.dense_layer_3(x)
        mask = F.softmax(mask, 1)
        att = x * mask
        output = torch.sum(att, 1)

        return self.dense_layer_4(output[:, None])


if __name__ == '__main__':
    sample_input = torch.ones((20, 12, 57))
    embedded_pts = torch.ones((20, 12, 91))

    AttNet = AttentionSets(input_ch=sample_input.shape[-1] + embedded_pts.shape[-1])
    out = AttNet(sample_input, embedded_pts)
    print(out.shape)
