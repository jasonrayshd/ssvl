import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x),  **kwargs)


class MLP(nn.Module):

    def __init__(self, dim, hidden_dim, dropout = 0., activation="gelu"):

        super(MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_hat =  self.ffn(x)

        return x_hat


class MHA(nn.Module):

    def __init__(self, dim, heads, dim_head, dropout=0.):

        super(MHA, self).__init__()

        hidden_dim = heads * dim_head

        self.heads = heads

        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, 3*hidden_dim, bias=False)

        self.scale = 1/math.sqrt(dim_head)

        project_out = not (heads == 1 and dim_head == dim)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        qkv = self.to_qkv(x).chunk(3, dim=-1) # shape of q/k/v (bs, patch num, heads*dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        scaled = F.softmax((q @ k.transpose(-1, -2))*self.scale, dim=2) 
        drop_scaled = self.dropout(scaled)
        y = drop_scaled @ v

        y = rearrange(y, 'b h n d -> b n (h d)')

        return self.to_out(y)


class EncoderLayer(nn.Module):

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0., norm=PreNorm):

        super(EncoderLayer, self).__init__()

        self.attn = norm(dim, MHA(dim = dim, heads=heads, dim_head=dim_head, dropout=dropout) )
        self.ffn = norm(dim, MLP(dim = dim, hidden_dim=mlp_dim, dropout=dropout) )


    def forward(self, x):

        x = self.attn(x) + x
        x = self.ffn(x) + x

        return x


class Encoder(nn.Module):

    def __init__(self, layers, dim, heads, dim_head, mlp_dim, dropout=0., norm=PreNorm):

        super(Encoder, self).__init__()

        self.norm = nn.LayerNorm(dim)
        Encoderlayers = []
        for i in range(layers):
            Encoderlayers.append(EncoderLayer(dim=dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout, norm=norm))

        self.encoder = nn.Sequential(*Encoderlayers)

    def forward(self, x):

        return self.encoder(x)

        # NOTE 2022.05.12 Jiachen Lei: 
        # In unofficial implementation of ViViT, github: https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py
        # the output of encoder is normalized by layernorm
        
        # return self.norm(self.encoder(x))


def pair(x):
    return x if isinstance(x, tuple) else (x, x)


class ViViT(nn.Module):

    def __init__(self, image_size, channels, num_classes, num_frames, patch_size, layers, dim, heads, dim_head=64, mlp_dim=256, dropout=0., emb_drop=0., norm=PreNorm, pool="cls", mode="share_learned"):
        """

            Parameters
            ---
            dim: dimension of patch token embedding

        """

        super(ViViT, self).__init__()

        assert pool in ["cls", "mean"], 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        patch_num = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        if mode == "share_learned":
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(patch_dim, dim),
            )
            self.position_embedding = nn.Parameter(torch.randn((1, num_frames, patch_num+1, dim)))
            self.dropout = nn.Dropout(emb_drop)

        elif mode == "sinusoid":
            pass
        else:
            raise ValueError(f"Specified mode:{mode} is not implemented")

        self.spatial_token= nn.Parameter(torch.randn((1, 1, dim)))
        self.temporal_token= nn.Parameter(torch.randn((1, 1, dim)))

        self.spatial_encoder = Encoder(layers=layers,
                                dim=dim, heads=heads, dim_head=dim_head,
                                mlp_dim=mlp_dim, dropout=dropout, norm=norm)

        self.temporal_encoder = Encoder(layers=layers,
                                dim=dim, heads=heads, dim_head=dim_head,
                                mlp_dim=mlp_dim, dropout=dropout, norm=norm)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )


    def forward(self, x):
        x = self.to_patch_embedding(x) # bs, tubelet, patch_num, dim
        b, t, n, _ = x.shape

        # spatial
        spatial_tokens = repeat(self.spatial_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((spatial_tokens, x), dim=2)
        x += self.position_embedding[:, :, :(n+1)]
        x = self.dropout(x)
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.spatial_encoder(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        # temporal
        temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((temporal_tokens, x), dim=1)
        x = self.temporal_encoder(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp_head(x)

        return x

def get_params(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters])
    print('Trainable Parameters: %.3fM' % parameters)


if __name__ == "__main__":
    import numpy as np
    img = torch.ones([1, 16, 3, 224, 224]).cuda()

    model = ViViT(image_size=224, channels=3, num_frames=16, num_classes=100, patch_size=16, layers=4, dim=192, heads=3, dim_head=64, mlp_dim=192*4).cuda()
    get_params(model)
    out = model(img)

    print("Shape of out :", out.shape)      # [B, num_classes]
