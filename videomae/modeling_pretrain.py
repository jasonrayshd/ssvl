import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from tokenizer_network import SimpleCNN

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_videomae_base_patch16_224', 
    'pretrain_videomae_large_patch16_224', 
]


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, num_frames=16,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_frames=num_frames, embed_dim=embed_dim,tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
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
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)
        
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        # print(x.shape)
        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
        # print(x_vis.shape)
        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2,
                 use_flow = False
                 ):
        super().__init__()
        self.num_classes = num_classes

        if not use_flow:
            assert num_classes == 3 * tubelet_size * patch_size ** 2

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
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
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=1536, #  decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm

                 use_flow=False,
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,

            use_flow = use_flow,
            )

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]

        return x


class Tokenizer(nn.Module):
    """
        Edited by jiachen
        Feature extractor
    """
    def __init__(self, in_chans, feature_dim, tubelet_size, patch_size:list,  backbone="single"):
        super().__init__()
        self.backbone = backbone

        if backbone == "single":
            self.tokenizer =  nn.Conv3d(in_channels=in_chans, out_channels=feature_dim, 
                            kernel_size = (tubelet_size,  patch_size[0], patch_size[1]), 
                            stride=(tubelet_size,  patch_size[0],  patch_size[1]))

        elif backbone == "simplecnn":
            self.tokenizer = SimpleCNN(in_ch=in_chans, tublet_size=tubelet_size, patch_size=patch_size, num_classes=feature_dim)

        else:
            raise NotImplementedError(f"Unkown tokenizer backbone: {backbone}, expected to be one of [single, simplecnn]")

    def forward(self, x, mask):
        if self.backbone == "single":
            x = self.tokenizer(x).flatten(2).transpose(1, 2)
        elif self.backbone == "simplecnn":
            x = self.tokenizer(x).flatten(2).transpose(1, 2)
        else:
            raise NotImplementedError(f"Unkown tokenizer backbone: {self.backbone}, expected to be one of [single, simplecnn]")

        B, _, C = x.shape
        x = x[~mask].reshape(B, -1, C)

        return x

class PretrainTsVisionTransformerSharedDecoder(nn.Module):
    """ 
        Edited by jiachen
        Two-stream vit shared decoder which reconstruct flow or image with separate heads
    """
    def __init__(self, rgb_num_classes=768, flow_num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, 
                #  use_flow = False
                 ):
        super().__init__()

        self.rgb_num_classes = rgb_num_classes
        self.flow_num_classes = flow_num_classes

        # if not use_flow:
        #     assert num_classes == 3 * tubelet_size * patch_size ** 2

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)

        self.rgb_head = nn.Linear(embed_dim, self.rgb_num_classes) if self.rgb_num_classes > 0 else nn.Identity()
        self.flow_head = nn.Linear(embed_dim, self.flow_num_classes) if self.flow_num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        raise NotImplementedError("no_weight_decay is not implemented in PretrainTsVisionTransformerSharedDecoder")

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        raise NotImplementedError("reset_classifier is not implemented in PretrainTsVisionTransformerSharedDecoder")
        # self.num_classes = num_classes
        # self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):

        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x_rgb = self.rgb_head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
            x_flow = self.flow_head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x_rgb = self.rgb_head(self.norm(x))
            x_flow = self.flow_head(self.norm(x))

        return x_rgb, x_flow

class PretrainTwoStreamVisionTransformer(nn.Module):
    """
        Edited by jiachen
        Two stream vision transformer
    """
    def __init__(self,
                img_size=224, 
                patch_size=16, 
                # encoder_in_chans=3, 
                encoder_num_classes=0, 
                encoder_embed_dim=768, 
                encoder_depth=12,
                encoder_num_heads=12, 

                rgb_num_classes = 1536, #  decoder_num_classes=768, 
                flow_num_classes = 512,

                decoder_embed_dim=512, 
                decoder_depth=8,
                decoder_num_heads=8, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                drop_rate=0., 
                attn_drop_rate=0.,
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm, 
                init_values=0.,
                use_learnable_pos_emb=False,
                tubelet_size=2,
                num_classes=0, # avoid the error from create_fn in timm
                in_chans=0, # avoid the error from create_fn in timm

                feature_dim = 768, # feature dimension of extracted features by tokenizer
                # share_tokenizer = False,
                # share_encoder = False,
                share_decoder = True,
                share_proj_layer = False,
                share_mask_token = False,
                # share_pos_embed = False,

                fuse_scheme = "concate",
                tokenizer_backbone = "I3DResNet",

                 ):
        super().__init__()

        # self.share_tokenizer = share_tokenizer
        # self.share_encoder = share_encoder
        self.share_decoder = share_decoder
        self.share_proj_layer = share_proj_layer
        self.share_mask_token =share_mask_token
        # self.share_pos_embed = share_pos_embed

        self.tokenizer_backbone = tokenizer_backbone
        self.fuse_scheme = fuse_scheme

        # tokenizer
        # if share_tokenizer:
        #     self.tokenizer = Tokenizer(in_chans, feature_dim, tubelet_size, patch_size, backbone=self.tokenizer_backbone)
        # else:
        self.rgb_tokenizer = Tokenizer(3, feature_dim, tubelet_size, [patch_size, patch_size], backbone=self.tokenizer_backbone)
        self.flow_tokenizer = Tokenizer(2, feature_dim, tubelet_size//2, [patch_size, patch_size], backbone=self.tokenizer_backbone)

        # encoder
        self.rgb_encoder = PretrainVisionTransformerEncoder(
                        img_size=img_size, 
                        patch_size=patch_size, 
                        in_chans=3, 
                        num_frames = 16,
                        num_classes=encoder_num_classes, 
                        embed_dim=encoder_embed_dim, 
                        depth=encoder_depth,
                        num_heads=encoder_num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale, 
                        drop_rate=drop_rate, 
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=drop_path_rate, 
                        norm_layer=norm_layer, 
                        init_values=init_values,
                        tubelet_size=tubelet_size,
                        use_learnable_pos_emb=use_learnable_pos_emb)

        self.flow_encoder = PretrainVisionTransformerEncoder(
                        img_size=img_size, 
                        patch_size=patch_size, 
                        in_chans=2, 
                        num_frames = 8,
                        num_classes=encoder_num_classes, 
                        embed_dim=encoder_embed_dim, 
                        depth=encoder_depth,
                        num_heads=encoder_num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale, 
                        drop_rate=drop_rate, 
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=drop_path_rate, 
                        norm_layer=norm_layer, 
                        init_values=init_values,
                        tubelet_size=tubelet_size//2,
                        use_learnable_pos_emb=use_learnable_pos_emb)

        # decoder
        if share_decoder:

            self.decoder = PretrainTsVisionTransformerSharedDecoder(

                            rgb_num_classes=rgb_num_classes, 
                            flow_num_classes=flow_num_classes,

                            embed_dim=decoder_embed_dim, 
                            depth=decoder_depth,
                            num_heads=decoder_num_heads, 
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, 
                            qk_scale=qk_scale, 
                            drop_rate=drop_rate, 
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=drop_path_rate, 
                            norm_layer=norm_layer, 
                            init_values=init_values,

                            )
        else:

            self.rgb_decoder = PretrainTsVisionTransformerSharedDecoder(

                            rgb_num_classes=rgb_num_classes, 
                            flow_num_classes=flow_num_classes,

                            embed_dim=decoder_embed_dim, 
                            depth=decoder_depth,
                            num_heads=decoder_num_heads, 
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, 
                            qk_scale=qk_scale, 
                            drop_rate=drop_rate, 
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=drop_path_rate, 
                            norm_layer=norm_layer, 
                            init_values=init_values,
                            )
            self.flow_decoder = PretrainTsVisionTransformerSharedDecoder(

                            rgb_num_classes=rgb_num_classes, 
                            flow_num_classes=flow_num_classes,

                            embed_dim=decoder_embed_dim, 
                            depth=decoder_depth,
                            num_heads=decoder_num_heads, 
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, 
                            qk_scale=qk_scale, 
                            drop_rate=drop_rate, 
                            attn_drop_rate=attn_drop_rate,
                            drop_path_rate=drop_path_rate, 
                            norm_layer=norm_layer, 
                            init_values=init_values,

                            )

        if share_proj_layer:
            self.encoder_to_decoder = nn.Linear(encoder_embed_dim + feature_dim, decoder_embed_dim, bias=False)
        else:
            self.rgb_encoder_to_decoder = nn.Linear(encoder_embed_dim + feature_dim, decoder_embed_dim, bias=False)
            self.flow_encoder_to_decoder = nn.Linear(encoder_embed_dim + feature_dim, decoder_embed_dim, bias=False)

        if share_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            trunc_normal_(self.mask_token, std=.02)
        else:
            self.rgb_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.flow_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            trunc_normal_(self.rgb_mask_token, std=.02)
            trunc_normal_(self.flow_mask_token, std=.02)

        # pos embedding is sinusoid, thus is not learnable
        # if share_pos_embed:
        #     self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)
        # else:
        self.rgb_pos_embed = get_sinusoid_encoding_table(self.rgb_encoder.patch_embed.num_patches, decoder_embed_dim)
        self.flow_pos_embed = get_sinusoid_encoding_table(self.flow_encoder.patch_embed.num_patches, decoder_embed_dim)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     result = set()
    #     if self.share_mask_token:
    #         result.add("mask_token")
    #     else:
    #         result.add("rgb_mask_token")
    #         result.add("flow_mask_token")
        
    #     if self.share_pos_embed:
    #         result.add("pos_embed")
    #     else:
    #         result.add("rgb_pos_embed")
    #         result.add("flow_pos_embed")
        
    #     return {'pos_embed', 'cls_token', 'mask_token'}

    def fuse(self, feat_e, feat_tok):
        if self.fuse_scheme == "concate":

            feat = torch.cat([feat_e, feat_tok], dim=2)
        else:
            raise NotImplementedError(f"Unknown fuse scheme:{self.fuse_scheme}, expected to be one of [concate, ]")

        return feat

    def forward(self, rgb, flows, mask):
        _, _, rgbT, _, _ = rgb.shape
        _, _, flowT, _, _ = flows.shape

        assert rgbT//flowT == 2, "number of rgb frames should be two times of flow images"

        # if self.share_encoder:
        #     rgb_vis = self.encoder(rgb, mask) # [B, N_vis, C_e]
        #     flow_vis = self.encoder(flows, mask) # [B, N_vis, C_e]
        # else:

        rgb_vis = self.rgb_encoder(rgb, mask) # [B, N_vis, C_e]
        flow_vis = self.flow_encoder(flows, mask) # [B, N_vis, C_e]
        # print(f"rgb_vis:{rgb_vis.shape} flow_vis:{flow_vis.shape}")

        # NOTE: different from encoder, input of tokenizer is unmasked rgb and flows
        # if self.share_tokenizer:
        #     rgb_token = self.tokenizer(rgb)         # [B, T, C_tok]
        #     flow_token = self.tokenizer(flows)      # [B, T, C_tok]
        # else:
        rgb_token = self.rgb_tokenizer(rgb, mask)     # [B, T, C_tok]
        flow_token = self.flow_tokenizer(flows, mask)  # [B, T, C_tok]
        # print(f"rgb_token:{rgb_token.shape} flow_token:{flow_token.shape}")

        # fuse and project
        rgb_feat = self.fuse(rgb_vis, rgb_token)
        flow_feat = self.fuse(flow_vis, flow_token)
        if self.share_proj_layer:
            rgb_feat = self.encoder_to_decoder(rgb_feat) # [B, N_vis, C_d]
            flow_feat = self.encoder_to_decoder(flow_feat) # [B, N_vis, C_d]
        else:
            rgb_feat = self.rgb_encoder_to_decoder(rgb_feat) # [B, N_vis, C_d]
            flow_feat = self.flow_encoder_to_decoder(flow_feat) # [B, N_vis, C_d]

        # print(f"rgb_feat:{rgb_feat.shape} flow_feat:{flow_feat.shape}")
        # print(f"mask: {mask.shape}")

        B, N, C = rgb_feat.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_rgb_pos_embed = self.rgb_pos_embed.expand(B, -1, -1).type_as(rgb_vis).to(rgb_vis.device).clone().detach()
        rgb_pos_emd_vis = expand_rgb_pos_embed[~mask].reshape(B, -1, C)
        rgb_pos_emd_mask = expand_rgb_pos_embed[mask].reshape(B, -1, C)
        # rgb_pos_emb = torch.cat([rgb_pos_emd_vis, rgb_pos_emd_mask], dim=1)
        # expand_rgb_mask_token = self.rgb_mask_token.expand_as(rgb_pos_emd_mask)
        # print(f"rgb_pos_emd_vis:{rgb_pos_emd_vis.shape}, rgb_pos_emd_mask:{rgb_pos_emd_mask.shape}")
        # print(f"expand_rgb_pos_embed:{expand_rgb_pos_embed.shape}")

        B, N, C = flow_feat.shape
        expand_flow_pos_embed = self.flow_pos_embed.expand(B, -1, -1).type_as(flow_vis).to(flow_vis.device).clone().detach()
        flow_pos_emd_vis = expand_flow_pos_embed[~mask].reshape(B, -1, C)
        flow_pos_emd_mask = expand_flow_pos_embed[mask].reshape(B, -1, C)
        # flow_pos_emb = torch.cat([flow_pos_emd_vis, flow_pos_emd_mask], dim=1)
        # expand_flow_mask_token = self.flow_mask_token.expand_as(flow_pos_emd_mask)
        # print(f"flow_pos_emd_vis:{flow_pos_emd_vis.shape}, flow_pos_emd_mask:{flow_pos_emd_mask.shape}")
        # print(f"expand_rgb_pos_embed:{expand_rgb_pos_embed.shape}")

        if self.share_mask_token:
            # assume same masking rate
            # assert rgb_vis.shape[1] == flow_vis.shape[1], f"rgb_vis:{rgb_vis.shape}, flow_vis:{flow_vis.shape}, expect rgb input and flow input have the same mask ratio"
            # message = f"expand_rgb_pos_embed:{expand_rgb_pos_embed.shape}, expand_flow_pos_embed:{expand_flow_pos_embed}, expect position embeding of rgb input and flow input have the same shape "
            # assert expand_rgb_pos_embed.shape == expand_flow_pos_embed.shape, message

            # expand_mask_token = self.mask_token.expand_as(flow_pos_emd_mask)
            rgb_full = torch.cat([rgb_feat + rgb_pos_emd_vis, self.mask_token + rgb_pos_emd_mask ], dim=1) # [B, N, C_d]
            flow_full = torch.cat([flow_feat + flow_pos_emd_vis, self.mask_token +flow_pos_emd_mask], dim=1) # [B, N, C_d]
        else:

            rgb_full = torch.cat([rgb_feat + rgb_pos_emd_vis, self.rgb_mask_token + rgb_pos_emd_mask], dim=1) # [B, N, C_d]
            flow_full = torch.cat([flow_feat + flow_pos_emd_vis, self.flow_mask_token + flow_pos_emd_mask], dim=1) # [B, N, C_d]

        # rgb_feat += expand_rgb_pos_embed
        # flow_feat += expand_flow_pos_embed
        # assert rgb_feat.shape == flow_feat.shape

        if self.share_decoder:
            rgb_rgb_hat, rgb_flow_hat = self.decoder(rgb_full, rgb_pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]
            flow_rgb_hat, flow_flow_hat = self.decoder(flow_full, flow_pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]
        else:
            rgb_rgb_hat, rgb_flow_hat = self.rgb_decoder(rgb_full, rgb_pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]
            flow_rgb_hat, flow_flow_hat = self.flow_decoder(flow_full, flow_pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]

        return rgb_rgb_hat, rgb_flow_hat, flow_rgb_hat, flow_flow_hat, rgb_vis, flow_vis, rgb_token, flow_token


@register_model
def pretrain_mae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=192, 
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 
 
@register_model
def pretrain_videomae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_videomae_flowA_small_patch16_224(pretrained=False, **kwargs):

    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=16*16*2*1, 
        decoder_embed_dim=192, 
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_flow=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_videomae_flowA_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=16*16*2*1, # patch_size * patch_size * number of flow axis * N
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_flow=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_tsvit_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainTwoStreamVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768,  # original: 768
        encoder_depth=12,        # original: 12
        encoder_num_heads=12,
        encoder_num_classes=0,
    
        rgb_num_classes=1536,
        flow_num_classes=16*16*2*1, # patch_size * patch_size * number of flow axis * N
        decoder_embed_dim=384,
        decoder_num_heads=6,
        # decoder_depth = 4,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),

        feature_dim = 768, # feature dimension of extracted features by tokenizer
        # share_tokenizer = False,
        # share_encoder = False,
        share_decoder = True,
        share_proj_layer = False,
        share_mask_token = False,
        # share_pos_embed = False,

        # fuse_scheme = "concate",
        # tokenizer_backbone = "I3DResNet",

        **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == "__main__":
    from masking_generator import TubeMaskingGenerator
    from engine_for_pretraining import TwoStreamVitLoss
    from einops import rearrange
    mask_gen = TubeMaskingGenerator(input_size=[8, 14, 14], mask_ratio=0.9)
    device = "cuda:1"
    B = 4
    model = pretrain_tsvit_base_patch16_224(decoder_depth = 4, tokenizer_backbone = "simplecnn").to(device)
    rgb = torch.randn((B, 3, 16, 224, 224)).to(device)
    flows = torch.randn((B, 2, 8, 224, 224)).to(device)
    mask = torch.from_numpy(mask_gen()).to(torch.bool).unsqueeze(0).to(device).repeat(B, 1)
    print(mask.shape)
    output = model(rgb, flows, mask)
    rgb_rgb_hat, rgb_flow_hat, flow_rgb_hat, flow_flow_hat, rgb_vis, flow_vis, rgb_token, flow_token = output
    print(rgb_rgb_hat.shape, rgb_flow_hat.shape, flow_rgb_hat.shape, flow_flow_hat.shape, rgb_vis.shape, flow_vis.shape, rgb_token.shape, flow_token.shape)
    loss_fn = TwoStreamVitLoss()

    videos_patch = rearrange(rgb, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=16, p2=16)
    B, _, C = videos_patch.shape
    rgb_target = videos_patch[mask].reshape(B, -1, C)

    B, _, N, H, W = flows.shape
    _, _, T, H, W = rgb.shape
    assert T%N == 0, f"Number of flows:{T} to be predicted should be divisible by number of frames:{N}"
    # print(flows.shape)

    flow_target = rearrange(flows, 'b c t (h p1) (w p2) -> b (t h w) (p1 p2 c)', p1=16, p2=16)

    tublet_size = 2
    bool_masked_pos_label = rearrange(mask, "b (t h w) -> b t h w", t=T//tublet_size, h=H//16,w=W//16)
    bool_masked_pos_label = bool_masked_pos_label.repeat(1, N//(T//tublet_size), 1, 1)
    bool_masked_pos_label = bool_masked_pos_label.reshape(B, -1)

    flow_target = flow_target[bool_masked_pos_label]
    flow_target = rearrange(flow_target, '(b t n) d -> b (t n) d', b=B, t=N//tublet_size)

    flow_target = flow_target.to(device, non_blocking=True)

    print(rgb_target.shape, flow_target.shape)


    loss_dct = loss_fn(output, [rgb_target, flow_target])

    loss_dct["sum"].backward()

    print(loss_dct)