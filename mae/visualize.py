import torch
from model_mae import mae_vit_base_patch16_dec512d8b



# model = mae_vit_base_patch16_dec512d8b()

ckpt_path = "mae_pretrain_vit_base.pth"

ckpt = torch.load(ckpt_path,  map_location="cpu")

print(ckpt["model"].keys())