import torch.nn as nn
import torch.nn.functional as F
import torch


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8,
                 hidden_dim=128, name=None, attention_output_length=256,
                 input_ch=3, embed_ch=3):
        super(SlotAttention, self).__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.dim = dim
        self.hidden_dim = max(dim, hidden_dim)

        self.input_ch = input_ch
        self.embed_ch = embed_ch
        self.attention_output_length = attention_output_length

        # 定义slot随机数tensor
        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, self.dim))

        # MLP
        self.mlp1 = nn.Linear(self.dim, self.dim)
        self.mlp2 = nn.Linear(self.dim, self.dim)

        # Dense
        self.dense = nn.Sequential(
            nn.Linear(self.input_ch, 64),  # dense1 relu
            nn.Linear(64, 32),  # dense2 relu
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
        )
        self.dense7 = nn.Linear(self.input_ch + self.embed_ch + 32, self.dim)

        # RNN
        self.gru = nn.GRUCell(self.dim, self.dim)

        # 定义attention的[Q,K,V]
        self.to_q = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.to_k = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.to_v = nn.Linear(in_features=self.dim, out_features=self.dim)

        # NORM
        self.norm_input = nn.LayerNorm(self.dim)
        self.norm_slots = nn.LayerNorm(self.dim)
        self.norm_pre_ff = nn.LayerNorm(self.dim)

    def forward(self, inputs, embed_pts, num_slots):
        max_rel = inputs
        for layer in self.dense:
            max_rel = layer(max_rel)

        inputs = torch.cat([inputs, embed_pts, max_rel], -1)
        inputs = self.dense7(inputs)

        batch, N, depth = inputs.shape
        num_slot = num_slots if num_slots is not None else self.num_slots

        mu = torch.broadcast_to(self.slots_mu, (batch, num_slot, self.dim))
        sigma = torch.broadcast_to(self.slots_sigma, (batch, num_slot, self.dim))
        slots = torch.normal(mean=torch.mean(mu).data, std=torch.std(sigma).data, size=mu.shape)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale  # Softmax(Q*K)*V/scale
            attn = nn.Softmax(1)(dots) + self.eps
            attn = attn / torch.sum(attn, -1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)
            gru_input = torch.reshape(updates, (-1, depth))
            gru_states = torch.reshape(slots_prev, (-1, depth))
            slots = self.gru(gru_input, gru_states)

            slots = torch.reshape(slots, (batch, -1, depth))  # (N,num_slot，dim)
            mlp_res = self.mlp1(self.norm_pre_ff(slots))
            mlp_res = self.mlp2(mlp_res)
            slots += mlp_res

        attention_out = torch.reshape(slots, [-1, num_slot * self.dim])

        return attention_out


if __name__ == '__main__':
    sample_input = torch.ones((20, 12, 57))
    indices = torch.ones((20, 12, 2))  # num_slot
    embedded_pts = torch.ones((20, 12, 91))

    slot_att = SlotAttention(4, 256, input_ch=57, embed_ch=91)
    output = slot_att(sample_input, embedded_pts, 4)

    print(output.shape)
