import os
from torchvision.utils import save_image
from torch.utils.data import Dataset
import glob
from PIL import Image
import numpy as np
import torch


def mkdir(cfg):
    os.makedirs(f'{cfg.base_dir}/{cfg.exp_name}/', exist_ok=True)
    os.makedirs(f'{cfg.base_dir}/{cfg.exp_name}/out/', exist_ok=True)
    os.makedirs(f'{cfg.base_dir}/{cfg.exp_name}/chkpt/', exist_ok=True)


def save_imgs(imgs, to_size, name) -> None:
    imgs = imgs.clamp(0, 1)
    imgs = imgs.view(imgs.size(0), *to_size)
    save_image(imgs, name)


class ImageFolder720p(Dataset):

    def __init__(self, folder_path):
        self.files = sorted(glob.glob('%s/*/*.*' % folder_path))

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        img = np.array(Image.open(path).resize((1280, 768), Image.ANTIALIAS))
        # h, w, c = img.shape

        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        patches = np.reshape(img, (3, 6, 128, 10, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))

        return img, patches, path

    def get_random(self):
        i = np.random.randint(0, len(self.files))
        return self[i]

    def __len__(self):
        return len(self.files)

