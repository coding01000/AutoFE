import os
from torchvision.utils import save_image


def mkdir(cfg):
    os.makedirs(f'{cfg.base_dir}/{cfg.exp_name}/', exist_ok=True)
    os.makedirs(f'{cfg.base_dir}/{cfg.exp_name}/out3/', exist_ok=True)
    os.makedirs(f'{cfg.base_dir}/{cfg.exp_name}/chkpt/', exist_ok=True)


def save_imgs(imgs, to_size, name) -> None:
    imgs = imgs.clamp(0, 1)
    imgs = imgs.view(imgs.size(0), *to_size)
    save_image(imgs, name)


