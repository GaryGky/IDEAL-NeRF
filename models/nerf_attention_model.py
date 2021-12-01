import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRFAttentionModel(nn.Module):
    def __init__(self, nerf_model, attention_model, attention_embed_ln):
        super(NeRFAttentionModel, self).__init__()
        self.nerf_model = nerf_model
        self.attention_model = attention_model
        self.attention_embed_ln = attention_embed_ln

    """
            Parameters
            ----------
            inputs[0]: embeded_pts (embed inputs + audio) dim=(2048*3)
            inputs[1]: cnn_features dim=(1024*64,196) cnn_feature(194) + indices(2)
            inputs[2[: input_flat (without embed)

            Returns
            -------
            NeRFAttentionModel prediction
    """

    def forward(self, inputs):
        """ n_pts, embedding_len """
        embeded_features = inputs[0]  # (netchunk,154 + 2278)
        """ indices: n_views, n_pts, 2; image_coords: n_views, n_pts, 2 """
        image_features = inputs[1]  # (netchunk,196)

        embeded_pts = embeded_features[..., :self.attention_embed_ln]  # (netchunk * 63)

        attention_inputs = torch.cat([image_features, embeded_pts], -1)

        # 把Attention模块的input_ch强制修改为 196+63
        self.attention_model.input_attn_ch = attention_inputs.shape[-1]

        attention_output = self.attention_model(attention_inputs)

        decoder_input = torch.cat([embeded_features, attention_output], -1)

        return self.nerf_model(decoder_input)
