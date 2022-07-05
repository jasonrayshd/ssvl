import gdown


def pretrain_ckpt_lst():

    ckpt = {
        "k400_videomae_pretrain_base_patch16_224_tubemasking_ratio_0.9_e800": "1JfrhN144Hdg7we213H1WxwR3lGYOlmIn",
        "k400_videomae_pretrain_base_patch16_224_tubemasking_ratio_0.9_e1600":"1tEhLyskjb755TJ65ptsrafUG2llSwQE1",
        "k400_videomae_pretrain_large_patch16_224_tubemasking_ratio_0.9_e1600": "1qLOXWb_MGEvaI7tvuAe94CV7S2HXRwT3",

        "ssv2_videomae_pretrain_base_patch16_224_tubemasking_ratio_0.9_epoch_800": "181hLvyrrPW2IOGA46fkxdJk0tNLIgdB2",
        "ssv2_videomae_pretrain_base_patch16_224_tubemasking_ratio_0.9_epoch_2400": "1I18dY_7rSalGL8fPWV82c0-foRUDzJJk",
    }

    return ckpt



def download_from(url, output):
    baseurl = "https://drive.google.com/uc?id="
    gdown.download(baseurl+url, output, quiet=False)
    
if __name__ == "__main__":
    file = "k400_videomae_pretrain_base_patch16_224_tubemasking_ratio_0.9_e800"
    download_from(pretrain_ckpt_lst()[file], f"./ckpt/{file}")