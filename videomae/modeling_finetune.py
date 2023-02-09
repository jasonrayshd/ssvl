from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from tokenizer_network import SimpleCNN, Tokenizer

import math
import einops

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,qk_scale=0, attn_head_dim=0):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
 
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

#         q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v)

#         x = x.transpose(1, 2).reshape(B, N, C)

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # print(in_chans, embed_dim, self.tubelet_size, patch_size)
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0], patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))


    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)

        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 keep_dim = False, # keep dimension of encoder extracted features or not ( will not return x[:,0] or x.mean(1) in forward_features() ))
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.keep_dim = keep_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.temporal_norm = norm_layer(embed_dim) if keep_dim else None

        self.head = nn.Linear(embed_dim, num_classes)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward_features(self, x):
        B, C, T, H, W = x.shape
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.keep_dim:
            num_patches = (H//self.patch_embed.patch_size[0]) * (W//self.patch_embed.patch_size[1])
            tubelet_size = self.patch_embed.tubelet_size
            x = einops.rearrange(x, "b (t n) d -> b t n d", t=T//tubelet_size, n=num_patches)
            x = x.mean(dim=2).squeeze()
            x = x.flatten(1)
            x = self.temporal_norm(x)
            return x

        if self.fc_norm is not None:
            return self.fc_norm( x.mean(1) )
        else:
            return self.norm(x)[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        # if self.keep_dim:
        #     x = x.permute(0, 2, 1)

        return x

class OSCCModel(nn.Module):

    def __init__(self, 
                img_size=224, 
                patch_size=16, 
                in_chans=3, 
                num_classes=1000, 
                embed_dim=768, 
                depth=12,
                num_heads=12, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm, 
                init_values=0.,
                use_learnable_pos_emb=False, 
                init_scale=0.,
                all_frames=16,
                tubelet_size=2,
                use_mean_pooling=True,
                # keep_dim = False,
                
                task = "oscc",
                ):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size

        self.task = task
        self.num_frames = all_frames
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # self.temporal_norm = norm_layer(embed_dim)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

        if self.task == "oscc":
            self.cls_head = nn.Linear(embed_dim, 2) 
        elif self.task == "pnr":
            # self.loc_head = nn.Linear(embed_dim, 2*embed_dim)
            self.cls_head = nn.Linear(embed_dim, self.num_frames+1)
        else:
            raise NotImplementedError(f"unknown task:{self.task}")

        trunc_normal_(self.cls_head.weight, std=.02)
        self.cls_head.weight.data.mul_(init_scale)
        self.cls_head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x):
        # input shape: B, C, T, H, W
        x = self.patch_embed(x) # patch embedding
        B, _, _ = x.size()
        # shape: B, T x patch_height x patch_width, hidden_dim
        # e.g. (32, 16/2 x 224/16 x 224/16, 768) -> (32, 1568, 768)

        # add positional embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x) # dropout
        # encoder
        for blk in self.blocks:
            x = blk(x) 

        return x


    def forward(self, x):
        # NOTE                            Jiachen 2022.05.25
        # from ego4d state change classification and localization
        # the raw tensor frames from __getitem__() shape like: C, T, H, W
        # Thus, if no transformations are used to augment x, the shape of x will be:
        # bs, C, T, H, W
        B, C, T, H, W = x.shape

        x = self.forward_features(x)
        # x shape: bs, T//tublet_size*patch num * patch num, embed_dim
        # if self.task == "oscc":
        if self.fc_norm is not None:
            pred = self.cls_head( self.fc_norm(x.mean(1)) )
        else:
            pred = self.cls_head( self.norm(x)[:, 0] )

        # elif self.task == "pnr":

        #     num_patches = (H//self.patch_embed.patch_size[0]) * (W//self.patch_embed.patch_size[1])
        #     tubelet_size = self.patch_embed.tubelet_size
        #     x = self.loc_head(x) # shape(b, m, 2*d)

        #     x = einops.rearrange(x, "b (T n) (t d) -> b (T t) n d", T=T//tubelet_size, t=tubelet_size, n=num_patches)

        #     x = x.mean(dim=2).squeeze(dim=2) # spatial pooling, shape (b,T,d)
        #     x = self.fc_norm(x)
        #     loc = self.cls_head(x) # shape (b, T, 1)
        #     loc = loc.squeeze()
        #     # loc = loc.permute(0, 2, 1) # for computing Cross-entropy loss

        return pred


class FintuneVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                img_size=224, 
                patch_size=16, 
                in_chans=3, 
                num_classes=1000, 
                embed_dim=768, 
                depth=12,
                num_heads=12, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm, 
                init_values=0.,
                use_learnable_pos_emb=False, 
                init_scale=0.,
                all_frames=16,
                tubelet_size=2,
                use_mean_pooling=True,
                keep_dim = False, # keep dimension of encoder extracted features or not ( will not return x[:,0] or x.mean(1) in forward_features() ))
                ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.keep_dim = keep_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        # self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        # self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # self.temporal_norm = norm_layer(embed_dim) if keep_dim else None

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward_features(self, x):
        B, C, T, H, W = x.shape
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x

class TwoHeadMultiCAE(nn.Module):

    def __init__(self, 
                img_size=224, 
                patch_size=16, 
                in_chans=3, 
                num_classes=1000, 
                embed_dim=768, 
                depth=12,
                num_heads=12, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm, 
                init_values=0.,
                use_learnable_pos_emb=False, 
                init_scale=0.,
                all_frames=16,
                tubelet_size=2,
                use_mean_pooling=True,
                # keep_dim = False,

                regressor_embed_dim = 384,
                regressor_num_heads = 6,
                regressor_depth = 4,
                
                ):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size

        self.num_frames = all_frames
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.regressor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, regressor_embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
            self.regressor_pos_embed = get_sinusoid_encoding_table(num_patches, regressor_embed_dim)
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.temporal_norm = norm_layer(embed_dim)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.regressor = MultiCAERegressor(
            patch_size=patch_size, 
            embed_dim=regressor_embed_dim, 

            depth=regressor_depth, 

            num_heads=regressor_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values, 
            num_patches=self.patch_embed.num_patches, 

        )
        self.flow_token = nn.Parameter(torch.zeros(1, 1, regressor_embed_dim))

        self.apply(self._init_weights)

        fc_dim = regressor_embed_dim*2
        self.proj = nn.Linear(embed_dim, regressor_embed_dim)
        self.cls_head = nn.Linear(fc_dim, 2)
        self.loc_head = nn.Linear(self.num_frames // self.tubelet_size * fc_dim, self.num_frames+1) # state change localization has num_frames+1

        trunc_normal_(self.cls_head.weight, std=.02)
        self.cls_head.weight.data.mul_(init_scale)
        self.cls_head.bias.data.mul_(init_scale)

        trunc_normal_(self.loc_head.weight, std=.02)
        self.loc_head.weight.data.mul_(init_scale)
        self.loc_head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x):
        # input shape: B, C, T, H, W
        x = self.patch_embed(x) # patch embedding
        B, _, _ = x.size()
        # shape: B, T x patch_height x patch_width, hidden_dim
        # e.g. (32, 16/2 x 224/16 x 224/16, 768) -> (32, 1568, 768)

        mask = torch.zeros_like(x)

        # add positional embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x) # dropout
        # encoder
        for blk in self.blocks:
            x = blk(x) 

        x = self.proj(x)

        expand_pos_embed = self.regressor_pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        flow_input = self.flow_token.expand(B, expand_pos_embed.shape[1], -1)
        flow_feat = self.regressor(flow_input, x, expand_pos_embed, expand_pos_embed,  mask)

        x = torch.cat([x, flow_feat], dim=2)

        return x


    def forward(self, x):
        # NOTE                            Jiachen 2022.05.25
        # from ego4d state change classification and localization
        # the raw tensor frames from __getitem__() shape like: C, T, H, W
        # Thus, if no transformations are used to augment x, the shape of x will be:
        # bs, C, T, H, W
        B, C, T, H, W = x.shape
        x = self.forward_features(x)
        # x shape: bs, T//tublet_size*patch num * patch num, embed_dim

        if self.fc_norm is not None:
            cls = self.cls_head( self.fc_norm(x.mean(1)) )
        else:
            cls = self.cls_head( self.norm(x)[:, 0] )

        num_patches = (H//self.patch_embed.patch_size[0]) * (W//self.patch_embed.patch_size[1])
        tubelet_size = self.patch_embed.tubelet_size
        x = einops.rearrange(x, "b (t n) d -> b t n d", t=T//tubelet_size, n=num_patches)
        x = x.mean(dim=2).squeeze(dim=2)
        x = self.temporal_norm(x)
        x = x.flatten(1)
        loc = self.loc_head(x) # shape: bs, frame num
        # loc = loc.permute(0, 2, 1) # for computing Cross-entropy loss

        return loc, cls


class Ego4dTwoHeadTwoStreamRGBVisionTransformer(nn.Module):

    def __init__(self, 
                img_size=224, 
                patch_size=16, 
                in_chans=3, 
                num_classes=1000, 
                embed_dim=768, 
                depth=12,
                num_heads=12, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm, 
                init_values=0.,
                use_learnable_pos_emb=False, 
                init_scale=0.,
                all_frames=16,
                tubelet_size=2,
                use_mean_pooling=True,

                ):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size

        self.num_frames = all_frames
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size//patch_size)* (img_size//patch_size) * (all_frames // tubelet_size)

        # if use_learnable_pos_emb:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        # else:
        #     # sine-cosine positional embeddings is on the way
        #     self.pos_embed = get_sinusoid_encoding_table(self.num_patches, embed_dim)

        # self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = [i for i in range(depth)]

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.rgb_encoder = FintuneVisionTransformerEncoder(
                                                img_size=img_size, 
                                                patch_size=patch_size, 
                                                in_chans=3, 
                                                # num_classes=1000, 
                                                embed_dim=embed_dim, 
                                                depth=depth,
                                                num_heads=num_heads, 
                                                mlp_ratio=mlp_ratio, 
                                                qkv_bias=qkv_bias, 
                                                qk_scale=qk_scale, 
                                                drop_rate=drop_rate, 
                                                attn_drop_rate=attn_drop_rate,
                                                drop_path_rate=drop_path_rate, 
                                                norm_layer=norm_layer, 
                                                init_values=init_values,
                                                use_learnable_pos_emb=use_learnable_pos_emb, 
                                                init_scale=init_scale,
                                                all_frames=all_frames,
                                                tubelet_size=tubelet_size,
                                                use_mean_pooling=use_mean_pooling,
                                        )

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.temporal_norm = norm_layer(embed_dim)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

        self.cls_head = nn.Linear(embed_dim, 2)
        self.loc_head = nn.Linear(self.num_frames // self.tubelet_size * embed_dim, self.num_frames+1) # state change localization has num_frames+1

        trunc_normal_(self.cls_head.weight, std=.02)
        self.cls_head.weight.data.mul_(init_scale)
        self.cls_head.bias.data.mul_(init_scale)

        trunc_normal_(self.loc_head.weight, std=.02)
        self.loc_head.weight.data.mul_(init_scale)
        self.loc_head.bias.data.mul_(init_scale)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed', "rgb_encoder.pos_embed",
            'cls_token', 
        }

    def forward_features(self, frames):

        flow_vis = self.rgb_encoder(frames)

        return flow_vis


    def forward(self, frames):
        # NOTE                            Jiachen 2022.05.25
        # from ego4d state change classification and localization
        # the raw tensor frames from __getitem__() shape like: C, T, H, W
        # Thus, if no transformations are used to augment x, the shape of x will be:
        # bs, C, T, H, W
        B, C, T, H, W = frames.shape
        x = self.forward_features(frames)

        # x = self.proj_head(x)
        # x shape: bs, T//tublet_size*patch num * patch num, embed_dim

        if self.fc_norm is not None:
            cls = self.cls_head( self.fc_norm(x.mean(1)) )
        else:
            cls = self.cls_head( self.norm(x)[:, 0] )

        num_patches = (H//self.patch_size[0]) * (W//self.patch_size[1])
        tubelet_size = self.tubelet_size
        x = einops.rearrange(x, "b (t n) d -> b t n d", t=T//tubelet_size, n=num_patches)
        # Simply using squeeze() without specifying dimension will raise a dimension mismatch error
        x = x.mean(dim=2).squeeze(dim=2)
        x = self.temporal_norm(x)
        x = x.flatten(1)
        loc = self.loc_head(x) # shape: bs, frame num
        # loc = loc.permute(0, 2, 1) # for computing Cross-entropy loss

        return loc, cls

class Ego4dTwoHeadTwoStreamwTokenzierVisionTransformer(nn.Module):

    def __init__(self, 
                img_size=224, 
                patch_size=16, 
                in_chans=3, 
                num_classes=1000, 
                embed_dim=768, 
                depth=12,
                num_heads=12, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm, 
                init_values=0.,
                use_learnable_pos_emb=False, 
                init_scale=0.,
                all_frames=16,
                tubelet_size=2,
                use_mean_pooling=True,

                tokenizer_backbone = "simplecnn"
                ):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size

        self.num_frames = all_frames
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size//patch_size)* (img_size//patch_size) * (all_frames // tubelet_size)

        self.tokenizer_backbone = tokenizer_backbone
        # if use_learnable_pos_emb:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        # else:
        #     # sine-cosine positional embeddings is on the way
        #     self.pos_embed = get_sinusoid_encoding_table(self.num_patches, embed_dim)

        # self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = [i for i in range(depth)]

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.rgb_encoder = FintuneVisionTransformerEncoder(
                                                img_size=img_size, 
                                                patch_size=patch_size, 
                                                in_chans=3, 
                                                # num_classes=1000, 
                                                embed_dim=embed_dim, 
                                                depth=depth,
                                                num_heads=num_heads, 
                                                mlp_ratio=mlp_ratio, 
                                                qkv_bias=qkv_bias, 
                                                qk_scale=qk_scale, 
                                                drop_rate=drop_rate, 
                                                attn_drop_rate=attn_drop_rate,
                                                drop_path_rate=drop_path_rate, 
                                                norm_layer=norm_layer, 
                                                init_values=init_values,
                                                use_learnable_pos_emb=use_learnable_pos_emb, 
                                                init_scale=init_scale,
                                                all_frames=all_frames,
                                                tubelet_size=tubelet_size,
                                                use_mean_pooling=use_mean_pooling,
                                        )

        self.rgb_tokenizer = Tokenizer(3, embed_dim, tubelet_size, [patch_size, patch_size], backbone=self.tokenizer_backbone)


        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.temporal_norm = norm_layer(embed_dim)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

        self.cls_head = nn.Linear(embed_dim, 2)
        self.loc_head = nn.Linear(self.num_frames // self.tubelet_size * embed_dim, self.num_frames+1) # state change localization has num_frames+1
        self.proj_head = nn.Linear(embed_dim*2, embed_dim)

        trunc_normal_(self.cls_head.weight, std=.02)
        self.cls_head.weight.data.mul_(init_scale)
        self.cls_head.bias.data.mul_(init_scale)

        trunc_normal_(self.loc_head.weight, std=.02)
        self.loc_head.weight.data.mul_(init_scale)
        self.loc_head.bias.data.mul_(init_scale)

        trunc_normal_(self.proj_head.weight, std=.02)
        self.proj_head.weight.data.mul_(init_scale)
        self.proj_head.bias.data.mul_(init_scale)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed', "rgb_encoder.pos_embed",
            'cls_token', 
        }

    def forward_features(self, frames):

        rgb_vis = self.rgb_tokenizer(frames)
        flow_vis = self.rgb_encoder(frames)

        feat = torch.cat([rgb_vis, flow_vis], dim=2)

        return feat


    def forward(self, frames):
        # NOTE                            Jiachen 2022.05.25
        # from ego4d state change classification and localization
        # the raw tensor frames from __getitem__() shape like: C, T, H, W
        # Thus, if no transformations are used to augment x, the shape of x will be:
        # bs, C, T, H, W
        B, C, T, H, W = frames.shape
        x = self.forward_features(frames)

        x = self.proj_head(x)
        # x shape: bs, T//tublet_size*patch num * patch num, embed_dim

        if self.fc_norm is not None:
            cls = self.cls_head( self.fc_norm(x.mean(1)) )
        else:
            cls = self.cls_head( self.norm(x)[:, 0] )

        num_patches = (H//self.patch_size[0]) * (W//self.patch_size[1])
        tubelet_size = self.tubelet_size
        x = einops.rearrange(x, "b (t n) d -> b t n d", t=T//tubelet_size, n=num_patches)
        # Simply using squeeze() without specifying dimension will raise a dimension mismatch error
        x = x.mean(dim=2).squeeze(dim=2)
        x = self.temporal_norm(x)
        x = x.flatten(1)
        loc = self.loc_head(x) # shape: bs, frame num
        # loc = loc.permute(0, 2, 1) # for computing Cross-entropy loss

        return loc, cls

class Ego4dTwoHeadTwoStreamCrossEncoderVisionTransformer(nn.Module):

    def __init__(self, 
                img_size=224, 
                patch_size=16, 
                in_chans=3, 
                num_classes=1000, 
                embed_dim=768, 
                depth=12,
                num_heads=12, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm, 
                init_values=0.,
                use_learnable_pos_emb=False, 
                init_scale=0.,
                all_frames=16,
                tubelet_size=2,
                use_mean_pooling=True,
                ):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size

        self.num_frames = all_frames
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size//patch_size)* (img_size//patch_size) * (all_frames // tubelet_size)

        # if use_learnable_pos_emb:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        # else:
        #     # sine-cosine positional embeddings is on the way
        #     self.pos_embed = get_sinusoid_encoding_table(self.num_patches, embed_dim)

        # self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = [i for i in range(depth)]

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.rgb_encoder = FintuneVisionTransformerEncoder(
                                                img_size=img_size, 
                                                patch_size=patch_size, 
                                                in_chans=3, 
                                                # num_classes=1000, 
                                                embed_dim=embed_dim, 
                                                depth=depth,
                                                num_heads=num_heads, 
                                                mlp_ratio=mlp_ratio, 
                                                qkv_bias=qkv_bias, 
                                                qk_scale=qk_scale, 
                                                drop_rate=drop_rate, 
                                                attn_drop_rate=attn_drop_rate,
                                                drop_path_rate=drop_path_rate, 
                                                norm_layer=norm_layer, 
                                                init_values=init_values,
                                                use_learnable_pos_emb=use_learnable_pos_emb, 
                                                init_scale=init_scale,
                                                all_frames=all_frames,
                                                tubelet_size=tubelet_size,
                                                use_mean_pooling=use_mean_pooling,
                                        )

        self.flow_encoder = FintuneVisionTransformerEncoder(
                                                img_size=img_size, 
                                                patch_size=patch_size, 
                                                in_chans=2, 
                                                # num_classes=1000, 
                                                embed_dim=embed_dim, 
                                                depth=depth,
                                                num_heads=num_heads, 
                                                mlp_ratio=mlp_ratio, 
                                                qkv_bias=qkv_bias, 
                                                qk_scale=qk_scale, 
                                                drop_rate=drop_rate, 
                                                attn_drop_rate=attn_drop_rate,
                                                drop_path_rate=drop_path_rate, 
                                                norm_layer=norm_layer, 
                                                init_values=init_values,
                                                use_learnable_pos_emb=use_learnable_pos_emb, 
                                                init_scale=init_scale,
                                                all_frames=all_frames//2,
                                                tubelet_size=tubelet_size//2,
                                                use_mean_pooling=use_mean_pooling,

        )


        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.temporal_norm = norm_layer(embed_dim)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

        self.cls_head = nn.Linear(embed_dim, 2)
        self.loc_head = nn.Linear(self.num_frames // self.tubelet_size * embed_dim, self.num_frames+1) # state change localization has num_frames+1
        self.proj_head = nn.Linear(embed_dim*2, embed_dim)

        trunc_normal_(self.cls_head.weight, std=.02)
        self.cls_head.weight.data.mul_(init_scale)
        self.cls_head.bias.data.mul_(init_scale)

        trunc_normal_(self.loc_head.weight, std=.02)
        self.loc_head.weight.data.mul_(init_scale)
        self.loc_head.bias.data.mul_(init_scale)

        trunc_normal_(self.proj_head.weight, std=.02)
        self.proj_head.weight.data.mul_(init_scale)
        self.proj_head.bias.data.mul_(init_scale)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed', "rgb_encoder.pos_embed", "flow_encoder.pos_embed",
            'cls_token', 
        }

    def forward_features(self, frames, flows):

        rgb_vis = self.rgb_encoder(frames)
        flow_vis = self.flow_encoder(flows)

        feat = torch.cat([rgb_vis, flow_vis], dim=2)

        return feat


    def forward(self, frames, flows):
        # NOTE                            Jiachen 2022.05.25
        # from ego4d state change classification and localization
        # the raw tensor frames from __getitem__() shape like: C, T, H, W
        # Thus, if no transformations are used to augment x, the shape of x will be:
        # bs, C, T, H, W
        B, C, T, H, W = frames.shape
        x = self.forward_features(frames, flows)

        x = self.proj_head(x)
        # x shape: bs, T//tublet_size*patch num * patch num, embed_dim

        if self.fc_norm is not None:
            cls = self.cls_head( self.fc_norm(x.mean(1)) )
        else:
            cls = self.cls_head( self.norm(x)[:, 0] )

        num_patches = (H//self.patch_size[0]) * (W//self.patch_size[1])
        tubelet_size = self.tubelet_size
        x = einops.rearrange(x, "b (t n) d -> b t n d", t=T//tubelet_size, n=num_patches)
        # Simply using squeeze() without specifying dimension will raise a dimension mismatch error
        x = x.mean(dim=2).squeeze(dim=2)
        x = self.temporal_norm(x)
        x = x.flatten(1)
        loc = self.loc_head(x) # shape: bs, frame num
        # loc = loc.permute(0, 2, 1) # for computing Cross-entropy loss

        return loc, cls

class Ego4dTwoHeadTwoStreamVisionTransformer(nn.Module):

    def __init__(self, 
                img_size=224, 
                patch_size=16, 
                in_chans=3, 
                num_classes=1000, 
                embed_dim=768, 
                depth=12,
                num_heads=12, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm, 
                init_values=0.,
                use_learnable_pos_emb=False, 
                init_scale=0.,
                all_frames=16,
                tubelet_size=2,
                use_mean_pooling=True,

                tokenizer_backbone="simplecnn"
                ):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size

        self.num_frames = all_frames
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size//patch_size)* (img_size//patch_size) * (all_frames // tubelet_size)

        # if use_learnable_pos_emb:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        # else:
        #     # sine-cosine positional embeddings is on the way
        #     self.pos_embed = get_sinusoid_encoding_table(self.num_patches, embed_dim)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        self.tokenizer_backbone = tokenizer_backbone
        self.blocks = [i for i in range(depth)]

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.rgb_encoder = FintuneVisionTransformerEncoder(
                                                img_size=img_size, 
                                                patch_size=patch_size, 
                                                in_chans=3, 
                                                # num_classes=1000, 
                                                embed_dim=embed_dim, 
                                                depth=depth,
                                                num_heads=num_heads, 
                                                mlp_ratio=mlp_ratio, 
                                                qkv_bias=qkv_bias, 
                                                qk_scale=qk_scale, 
                                                drop_rate=drop_rate, 
                                                attn_drop_rate=attn_drop_rate,
                                                drop_path_rate=drop_path_rate, 
                                                norm_layer=norm_layer, 
                                                init_values=init_values,
                                                use_learnable_pos_emb=use_learnable_pos_emb, 
                                                init_scale=init_scale,
                                                all_frames=all_frames,
                                                tubelet_size=tubelet_size,
                                                use_mean_pooling=use_mean_pooling,
                                        )

        self.flow_encoder = FintuneVisionTransformerEncoder(
                                                img_size=img_size, 
                                                patch_size=patch_size, 
                                                in_chans=2, 
                                                # num_classes=1000, 
                                                embed_dim=embed_dim, 
                                                depth=depth,
                                                num_heads=num_heads, 
                                                mlp_ratio=mlp_ratio, 
                                                qkv_bias=qkv_bias, 
                                                qk_scale=qk_scale, 
                                                drop_rate=drop_rate, 
                                                attn_drop_rate=attn_drop_rate,
                                                drop_path_rate=drop_path_rate, 
                                                norm_layer=norm_layer, 
                                                init_values=init_values,
                                                use_learnable_pos_emb=use_learnable_pos_emb, 
                                                init_scale=init_scale,
                                                all_frames=all_frames//2,
                                                tubelet_size=tubelet_size//2,
                                                use_mean_pooling=use_mean_pooling,

        )

        self.rgb_tokenizer = Tokenizer(3, embed_dim, tubelet_size, [patch_size, patch_size], backbone=self.tokenizer_backbone)
        self.flow_tokenizer =Tokenizer(2, embed_dim, tubelet_size//2, [patch_size, patch_size], backbone=self.tokenizer_backbone)

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.temporal_norm = norm_layer(embed_dim)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

        self.cls_head = nn.Linear(embed_dim, 2)
        self.loc_head = nn.Linear(self.num_frames // self.tubelet_size * embed_dim, self.num_frames+1) # state change localization has num_frames+1
        # project to embed_dim
        self.proj_head = nn.Linear(embed_dim*2, embed_dim)

        # fuse information from rgb and flow respectively
        self.fuse_head = nn.Linear(embed_dim*2, embed_dim)

        trunc_normal_(self.cls_head.weight, std=.02)
        self.cls_head.weight.data.mul_(init_scale)
        self.cls_head.bias.data.mul_(init_scale)

        trunc_normal_(self.loc_head.weight, std=.02)
        self.loc_head.weight.data.mul_(init_scale)
        self.loc_head.bias.data.mul_(init_scale)

        trunc_normal_(self.proj_head.weight, std=.02)
        self.proj_head.weight.data.mul_(init_scale)
        self.proj_head.bias.data.mul_(init_scale)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed', "rgb_encoder.pos_embed", "flow_encoder.pos_embed",
            'cls_token', 
        }

    def forward_features(self, frames, flows):

        rgb_vis = self.rgb_encoder(frames)
        rgb_feat = self.rgb_tokenizer(frames)

        flow_vis = self.flow_encoder(flows)
        flow_feat = self.flow_tokenizer(flows)

        rgb_cat = torch.cat([rgb_vis, rgb_feat], dim=2)
        flow_cat = torch.cat([flow_feat, flow_vis], dim=2)

        return rgb_cat, flow_cat


    def forward(self, frames, flows):
        # NOTE                            Jiachen 2022.05.25
        # from ego4d state change classification and localization
        # the raw tensor frames from __getitem__() shape like: C, T, H, W
        # Thus, if no transformations are used to augment x, the shape of x will be:
        # bs, C, T, H, W
        B, C, T, H, W = frames.shape
        rgb_cat, flow_cat = self.forward_features(frames, flows)

        rgb_proj = self.proj_head(rgb_cat)
        flow_proj = self.proj_head(flow_cat)

        x = self.fuse_head(torch.cat([rgb_proj, flow_proj], dim=2))

        # x shape: bs, T//tublet_size*patch num * patch num, embed_dim

        if self.fc_norm is not None:
            cls = self.cls_head( self.fc_norm(x.mean(1)) )
        else:
            cls = self.cls_head( self.norm(x)[:, 0] )

        num_patches = (H//self.patch_size[0]) * (W//self.patch_size[1])
        tubelet_size = self.tubelet_size
        x = einops.rearrange(x, "b (t n) d -> b t n d", t=T//tubelet_size, n=num_patches)
        # Simply using squeeze() without specifying dimension will raise a dimension mismatch error
        x = x.mean(dim=2).squeeze(dim=2)
        x = self.temporal_norm(x)
        x = x.flatten(1)
        loc = self.loc_head(x) # shape: bs, frame num
        # loc = loc.permute(0, 2, 1) # for computing Cross-entropy loss

        return loc, cls


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, bool_masked_pos=None, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)    # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)   
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))      # (B, N_head, N_q, N_k)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) 
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class RegressorBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.norm2_cross = norm_layer(dim)
        self.cross_attn =  CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp_cross = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1_cross = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2_cross = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1_cross = nn.Parameter(torch.ones((dim)),requires_grad=False)
            self.gamma_2_cross = nn.Parameter(torch.ones((dim)),requires_grad=False)

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos):
        x = x_q + self.drop_path(self.gamma_1_cross * self.cross_attn(self.norm1_q(x_q + pos_q),
         bool_masked_pos, k=self.norm1_k(x_kv + pos_k), v=self.norm1_v(x_kv)))
        x = self.norm2_cross(x)
        x = x + self.drop_path(self.gamma_2_cross * self.mlp_cross(x))

        return x


class MultiCAERegressor(nn.Module):
    def __init__(self, patch_size=16, num_classes=8192, embed_dim=768, depth=6, 
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, num_patches=196, init_std=0.02, args=None, patch_shape=(14,14)):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.args = args

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # context regressor
        self.regressor_blocks = nn.ModuleList([
            RegressorBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.init_std = init_std

        # init the model
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.regressor_blocks):
            rescale(layer.cross_attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp_cross.fc2.weight.data, layer_id + 1)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked, bool_masked_pos):                
        # latent contextual regressor
        for blk in self.regressor_blocks:
            x_masked = blk(x_masked, torch.cat([x_unmasked, x_masked], dim=1), pos_embed_masked, torch.cat([pos_embed_unmasked, pos_embed_masked], dim=1), bool_masked_pos)
        x_masked = self.norm(x_masked)

        return x_masked


class VarAnt(nn.Module):
    def __init__(self, 
            hidden_dim = 256,     # hidden dimension for RNNs
            latent_dim = 128,    # latent dimension for goal
            num_act_cand = 10,   # number of action candidates used for nextiction
            n_layers = 1,        # number of layers in sampler RNN for generating next action latent goal
            num_goal_cand = 3,   # number of samples drawn from observed goal distribution
            next_num = 3,        # number of anticipated actions defined in Ego4d long-term anticipation
            num_classes = 1000,  # number of output classes 
            feat_dim = 256,        # dimension of input sequential feature
            drop_rate = 0.2,
        ):

        super(VarAnt, self).__init__()

        self.feat_dim = feat_dim
        self.h_dim = hidden_dim 
        self.z_dim = latent_dim 
        self.num_act_cand = num_act_cand
        self.n_layers = n_layers
        self.num_goal_cand = num_goal_cand
        self.next_num = next_num
        self.num_classes = num_classes

        # # # Linear layer for observed feature x_t
        self.phi_x = nn.Sequential(nn.Linear(self.feat_dim, self.h_dim),
                                   nn.ReLU())
        # # # Linear layer for encoder - observed feature x_t and hidden state from t-1
        self.phi_enc = nn.Sequential(nn.Linear(self.h_dim + self.h_dim, self.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.h_dim, self.z_dim))
        # # #  MLP for prior - input is hidden state from t-1
        self.phi_prior = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.h_dim, self.z_dim))
        # # #  Linear layer for latent goal z_t at time t
        self.phi_z = nn.Sequential(nn.Linear(self.z_dim, self.h_dim),
                                   nn.ReLU())

        # # # RNN to generate prior and encoder distributions for observed features
        self.rnn = nn.GRU(2*self.h_dim, self.h_dim, 1)
        
        # # # After getting the last latent goal change phi_z for sample
        self.phi_z_new = nn.Sequential(nn.Linear(self.z_dim, self.h_dim),
                                   nn.ReLU())
        # # # After getting the last layer change phi_prior 
        self.phi_prior_new = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.h_dim, self.z_dim))
        # # # Layer normalization on rnn output
        # self.layernorm = nn.LayerNorm(self.h_dim)
        
        # # # Linear layer for Observed action based on latent goal and final hidden state 
        # # # OR
        # # # Linear layer for nexticted action candidate based on observed action and final hidden state 
        self.phi_get_act = nn.Sequential(nn.Linear(self.z_dim + self.h_dim, self.h_dim),
                                   nn.ReLU())

        # # #  RNN to generate nexticted visual feature for nexticted latent goal
        # self.sampler_rnn = nn.GRU(h_dim, h_dim, self.n_layers)
        
        # # # Linear layer for nexticted latent goal 
        # self.phi_prior_next = nn.Sequential(nn.Linear(h_dim, z_dim), nn.ReLU())

        # # # Linear layer to transform observed action to z_dim each
        self.phi_obs_act = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.h_dim, self.z_dim))
                                       
        # # # Linear layer to transform next action to z_dim
        self.phi_next_act = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.h_dim, self.z_dim))


        # # # Linear layer for next latent goal - inputs are next and observed action
        self.phi_enc_next = nn.Sequential(nn.Linear(2*self.z_dim, self.z_dim), nn.ReLU())

        self.phi_act = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                       nn.ReLU(),)

        # rnn for action anticipation
        # hidden state: h_dim
        # input: action prior(h_dim) and current action()
        self.action_rnn = nn.GRU(2*self.h_dim, self.h_dim, 1)

        # # # For generating std of distributions
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop_rate)

        self.cur_act_classifier = nn.Linear(self.h_dim, self.num_classes)
        self.next_act_classifier = nn.Linear(self.h_dim, self.num_classes)

        # self.act_classifier_lst = nn.ModuleList([
        #     nn.Linear(self.h_dim, self.num_classes)
        #     for i in range(self.next_num-1)
        # ])

    def init_hidden(self, x):
        return torch.randn(self.n_layers, x.size(1), self.h_dim).to(x)

    def reparameterized_sample(self, mean, std, device):
        """using std to sample"""
        eps = torch.randn(std.shape, requires_grad=True).to(device)
        return eps.mul(std).add_(mean)

    def kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return .5 * torch.sum(kld_element)
    
    def sym_kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element_1 = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
                       
        kld_element_2 = (2 * torch.log(std_1) - 2 * torch.log(std_2) +
                       (std_2.pow(2) + (mean_2 - mean_1).pow(2)) /
                       std_1.pow(2) - 1)
        return .5 * (torch.sum(kld_element_1) + torch.sum(kld_element_2))

    def forward(self, feat_seq, batch_size=None):

        self.rnn.flatten_parameters()
        self.action_rnn.flatten_parameters()

        kld_lat_goal = 0 
        if len(feat_seq.shape) == 2:
            feat_seq = feat_seq.unsqueeze(0)
        h = self.init_hidden(feat_seq)
        # # # conditional VRNN for generating latent goal based on observed features

        for t in range(feat_seq.shape[0]):
            # print('feat dim:',self.feat_dim)
            x_t = feat_seq[t,:]
            # print('x_t:',x_t.shape)
            # # # Encoder distribution that combines observed feature at time t and hidden state from t-1
            enc_t = self.phi_enc(torch.cat([h[-1], self.phi_x(x_t)], -1))

            enc_mean_t = enc_t
            enc_std_t = self.softplus(enc_t)

            # # # Prior distribution based on hidden state from t-1
            prior_t = self.phi_prior(h[-1])
            prior_mean_t = prior_t
            prior_std_t = self.softplus(prior_t)

            # # # Comparing KLDiv between encoder and prior distributions
            kld_lat_goal += self.kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

            # # # Latent goal at time t
            z_t = self.reparameterized_sample(enc_mean_t, enc_std_t, device = feat_seq.device).type_as(feat_seq)

            _, h = self.rnn(torch.cat([self.phi_x(x_t).unsqueeze(0), self.phi_z(z_t).unsqueeze(0)], -1), h)

            # # #  applying layernorm on hidden state
            # h = self.layernorm(h)
        # # # Distribution for latent goal for observed sequence
        obs_lat_goal_mean = self.phi_prior(h[-1])
        obs_lat_goal_std = self.softplus(self.phi_prior(h[-1]))

        lat_goal_dis = 1e9
        next_act_final = 0.
        obs_act_final = 0.
        kld_next_lat_goal = 0.
        for _ in range(self.num_goal_cand):
            # # # Sample latent goal for observed sequence
            obs_lat_goal = self.reparameterized_sample(obs_lat_goal_mean, obs_lat_goal_std, device = feat_seq.device).type_as(feat_seq)
            # # # Observed action based on latent goal and final hidden state 
            obs_act = self.phi_get_act(torch.cat([self.phi_z_new(obs_lat_goal).unsqueeze(0), self.phi_prior_new(h[-1]).unsqueeze(0)], -1))

            # # # Predicted(next) action using observed action and final hidden state
            next_act_dummy = self.phi_get_act(torch.cat([self.phi_prior_new(h[-1]).unsqueeze(0), obs_act], -1))
            next_act_mean = next_act_dummy
            next_act_std = self.softplus(next_act_dummy)

            for i in range(self.num_act_cand): 
                # # # Next action sample
                next_act = self.reparameterized_sample(next_act_mean, next_act_std, device = feat_seq.device).type_as(feat_seq)
                # print('next_act:', next_act.shape)
                # print('obs_act:', obs_act.shape)

                # CVAE for next action 
                # # # Prior distribution for next latent goal
                prior_next_lat_goal_mean = self.phi_next_act(next_act)
                prior_next_lat_goal_std = self.softplus(self.phi_next_act(next_act))
                # print('prior_next_lat_goal_std:', prior_next_lat_goal_std.shape)

                # # # Encoder distribution for nexticted latent goal conditioned on observed action
                enc_next_lat_goal = self.phi_enc_next(torch.cat([self.phi_next_act(next_act), self.phi_obs_act(obs_act)], -1))
                enc_next_lat_goal_mean = enc_next_lat_goal
                enc_next_lat_goal_std = self.softplus(enc_next_lat_goal)
                # print('enc_next_lat_goal_std:', enc_next_lat_goal_std.shape)

                # # # Symmetric KLDiv between latent goal distribution for observed action and nexticted action
                new_lat_dis = self.sym_kld_gauss( obs_lat_goal_mean, obs_lat_goal_std, \
                                                enc_next_lat_goal_mean, enc_next_lat_goal_std)

                # if i == 0:
                #     lat_goal_dis = new_lat_dis
                #     next_act_final = next_act
                #     obs_act_final = obs_act
                #     kld_next_lat_goal = self.kld_gauss( enc_next_lat_goal_mean, enc_next_lat_goal_std,\
                #                                prior_next_lat_goal_mean, prior_next_lat_goal_std)
                # # # The action candidate with minimal Symmetric KLD is chosen
                if i == 0 or new_lat_dis < lat_goal_dis:
                    
                    lat_goal_dis = new_lat_dis
                    # # # Comparing KLDiv between encoder and prior distributions of nexticted latent goal
                    kld_next_lat_goal = self.kld_gauss( enc_next_lat_goal_mean, enc_next_lat_goal_std,\
                                            prior_next_lat_goal_mean, prior_next_lat_goal_std)
                    # # # Predicted action
                    next_act_final = next_act
                    obs_act_final = obs_act

        # found best observed action and next action pair
        # keep predicting future actions
        kld_future_lat_goal = [0. for i in range(self.next_num-1)]
        future_act_final = [0. for i in range(self.next_num-1)]
        future_goal_dis = [1e9 for i in range(self.next_num-1)]

        pre_act = obs_act_final # previous action
        cur_act = next_act_final # current action
        # init hidden state for action anticipation
        h_action = h.clone() # initialize first action hidden state with h_T

        for i in range(self.next_num-1):

            _, h_action = self.action_rnn(torch.cat([self.phi_act(pre_act), self.phi_act(cur_act)], -1), h_action)

            pre_act = cur_act

            future_act_dummy = self.phi_get_act(torch.cat([self.phi_prior_new(h_action[-1]).unsqueeze(0), pre_act], -1))
            future_act_mean = future_act_dummy
            future_act_std = self.softplus(future_act_dummy)

            for j in range(self.num_act_cand): 
                # # # Next action sample
                future_act = self.reparameterized_sample(future_act_mean, future_act_std, device = feat_seq.device).type_as(feat_seq)

                prior_future_lat_goal_mean = self.phi_next_act(future_act)
                prior_future_lat_goal_std = self.softplus(self.phi_next_act(future_act))
                # print('prior_next_lat_goal_std:', prior_next_lat_goal_std.shape)

                # # # Encoder distribution for nexticted latent goal conditioned on observed action
                enc_future_lat_goal = self.phi_enc_next(torch.cat([self.phi_next_act(future_act), self.phi_next_act(pre_act)], -1))
                enc_future_lat_goal_mean = enc_future_lat_goal
                enc_future_lat_goal_std = self.softplus(enc_future_lat_goal)
                # print('enc_next_lat_goal_std:', enc_next_lat_goal_std.shape)

                # # # Symmetric KLDiv between latent goal distribution for observed action and nexticted action
                new_lat_dis = self.sym_kld_gauss( obs_lat_goal_mean, obs_lat_goal_std, \
                                                enc_future_lat_goal_mean, enc_future_lat_goal_std)

                if j == 0 or new_lat_dis < future_goal_dis[i]:
                    future_goal_dis[i] = new_lat_dis

                    kld_future_lat_goal[i] = self.kld_gauss( enc_future_lat_goal_mean, enc_future_lat_goal_std,\
                                            prior_future_lat_goal_mean, prior_future_lat_goal_std)
                    # # # Predicted action
                    future_act_final[i] = future_act

            cur_act = future_act_final[i]

        # # # Embedding both next and current action to number of action classes
        try:
            out_next = self.next_act_classifier(next_act_final).squeeze(0)
            out_future = [self.next_act_classifier(future_act_final[i]).squeeze(0) for i in range(self.next_num-1)]  
            out_future = [out_next, *out_future]
        except:
            print('next_act_final:', next_act_final.shape)

        out_cur = self.cur_act_classifier(obs_act_final)

        return out_cur.squeeze(0), out_future, kld_lat_goal, kld_next_lat_goal, lat_goal_dis, \
             sum(kld_future_lat_goal), sum(future_goal_dis)


class MultiTaskHead(nn.Module):
    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        super(MultiTaskHead, self).__init__()

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.

        self.projection = nn.Linear(dim_in, num_classes, bias=True)
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, feat):
        """
            Args:
                feat : (B, T, H, W, C)
        """

        if hasattr(self, "dropout"):
            feat = self.dropout(feat)
        B,*_, C = feat.shape
        feat = feat.reshape(B, -1, C).mean(1)

        feat = self.projection(feat)

        return feat


class LongTermActionAnticipationModel(nn.Module):
    """ 
        Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                img_size=224, 
                patch_size=16, 
                in_chans=3, 
                num_classes=1000, 
                embed_dim=768, 
                depth=12,
                num_heads=12, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm, 
                init_values=0.,
                use_learnable_pos_emb=False, 
                init_scale=0.,
                all_frames=16,
                tubelet_size=2,
                use_mean_pooling=True,
                keep_dim = False, # keep dimension of encoder extracted features or not ( will not return x[:,0] or x.mean(1) in forward_features() ))
                
                head_type = "baseline"
                ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.keep_dim = keep_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.head_type = head_type

        if self.head_type == "varant":
            self.anticipation_model = VarAnt(

                num_classes = num_classes,
                next_num = 20,
                drop_rate = 0.2,
                feat_dim=embed_dim,  # dimension of input sequential feature
                hidden_dim = 512,     # hidden dimension for RNNs
                latent_dim = 256,    # latent dimension for goal
                num_act_cand = 10,   # number of action candidates used for nextiction
                n_layers = 1,        # number of layers in sampler RNN for generating next action latent goal
                num_goal_cand = 3,   # number of samples drawn from observed goal distribution
            )
        elif self.head_type == "baseline":
             self.anticipation_model = nn.ModuleList([
                    MultiTaskHead(
                    dim_in=embed_dim,
                    num_classes = num_classes,
                    dropout_rate=0.1,
                    act_func="softmax",
                ) for i in range(21)
             ])

        self.apply(self._init_weights)
        # self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        # self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # self.temporal_norm = norm_layer(embed_dim) if keep_dim else None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward_features(self, x):
        B, C, T, H, W = x.shape
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        # x: B, C, T, H, W,
        B, C, T, H, W = x.shape

        x = self.forward_features(x)
        # x: B, (T//t)*(H//p0)*(W//p1), p0*p1*t*c
        if self.head_type == "varant":
            feat_seq = einops.rearrange(x, "b (t h w) c -> t b (h w) c", h=H//self.patch_size, w=W//self.patch_size,t=T//self.tubelet_size,)
            feat_seq = feat_seq.mean(2).squeeze(2) # T B C
            out = self.anticipation_model(feat_seq)
        elif self.head_type == "baseline":
            feat_seq = einops.rearrange(x, "b (t h w) c -> b t h w c", h=H//self.patch_size, w=W//self.patch_size,t=T//self.tubelet_size,)
            out = [self.anticipation_model[i](feat_seq) for i in range(21)]

        # out = out_cur, out_future, kld_lat_goal, kld_next_lat_goal, lat_goal_dis, kld_future_lat_goal, future_goal_dis
        return out


class HandsPredictionHead(nn.Module):

    def __init__(self, 
        embed_dim = 256,  # model hidden dimension
        feat_dim = 768,    # input feature dimension

        drop_rate = 0.3,
        method = "baseline"):

        super().__init__()

        self.method = method
        self.embed_dim = embed_dim

        if method == "baseline":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            self.norm = norm_layer(feat_dim)
            # self.dropout = nn.Dropout(drop_rate)
            self.head = nn.Sequential(
                nn.Linear(feat_dim,embed_dim),
                nn.Dropout(drop_rate),
                nn.ReLU(),
                nn.Linear(embed_dim,20),   
            )

    def forward(self, x):
        
        if self.method == "baseline":
            # x: B, N, C
            x = self.norm(x)
            x = x.mean(1).squeeze()
            preds = self.head(x)
            return preds


class FutureHandsPredictionModel(nn.Module):
    """ 
        Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                img_size=224, 
                patch_size=16, 
                in_chans=3, 
                num_classes=1000, 
                embed_dim=768, 
                depth=12,
                num_heads=12, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm, 
                init_values=0.,
                use_learnable_pos_emb=False, 
                init_scale=0.,
                all_frames=16,
                tubelet_size=2,
                use_mean_pooling=True,
                keep_dim = False, # keep dimension of encoder extracted features or not ( will not return x[:,0] or x.mean(1) in forward_features() ))
                ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.keep_dim = keep_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        # self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        # self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # self.temporal_norm = norm_layer(embed_dim) if keep_dim else None

        self.hands_prediction_head = HandsPredictionHead(embed_dim=256,feat_dim=embed_dim, method="baseline")

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward_features(self, x):
        B, C, T, H, W = x.shape
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        # x: B, C, T, H, W,
        B, C, T, H, W = x.shape
        x = self.forward_features(x)
        # x: B, (T//t)*(H//p0)*(W//p1), p0*p1*t*c

        out = self.hands_prediction_head(x)

        return out



@register_model
def vit_egoclip_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    return model


@register_model
def vit_hands_base_patch16_224(pretrained=False, **kwargs):
    model = FutureHandsPredictionModel(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    return model

@register_model
def vit_hands_base_patch16_320(pretrained=False, **kwargs):
    model = FutureHandsPredictionModel(
        img_size=320, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    return model

@register_model
def vit_lta_base_patch16_224(pretrained=False, **kwargs):
    model = LongTermActionAnticipationModel(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    return model


@register_model
def vit_twohead_multicae_base_patch16_224(pretrained=False, **kwargs):
    model = TwoHeadMultiCAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    return model


@register_model
def oscc_vit_base_patch16_224(pretrained=False, **kwargs):

    model = OSCCModel(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()
    return model

@register_model
def oscc_vit_large_patch16_224(pretrained=False, **kwargs):

    model = OSCCModel(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()
    return model


@register_model
def vit_twohead_ts_rgb_cross_encoder_base_patch16_224(pretrained=False, **kwargs):

    model = Ego4dTwoHeadTwoStreamRGBVisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()
    return model

@register_model
def vit_twohead_ts_rgb_base_patch16_224(pretrained=False, **kwargs):

    model = Ego4dTwoHeadTwoStreamwTokenzierVisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()
    return model

@register_model
def vit_twohead_ts_cross_encoder_base_patch16_224(pretrained=False, **kwargs):

    model = Ego4dTwoHeadTwoStreamCrossEncoderVisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()
    return model

@register_model
def vit_twohead_ts_base_patch16_224(pretrained=False, **kwargs):

    model = Ego4dTwoHeadTwoStreamVisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()
    return model

@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):

    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()
    return model

@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):

    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model



if __name__ == "__main__":
    device = "cuda:0"
    model = LongTermActionAnticipationModel(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        
        num_classes= 1000,
        all_frames= 16,
        tubelet_size=2,
        drop_rate=0,
        drop_path_rate=0,
        attn_drop_rate=0,
        use_mean_pooling=True,
        init_scale=0.001,
        ).to(device)

    # x = torch.randn((4,3,16,224,224)).to(device)

    # out = model(x)
