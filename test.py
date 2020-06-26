import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import utils
import mylayer
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

cfg = {
    "num_epochs": 5,
    "start_epoch": 1,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "exp_name": "Image_Compressiong",
    "base_dir": "/content/drive/My Drive",
    "save_every": 200,
    "shuffle": False,
    "dataset_path": "/content/drive/My Drive/NWPU",
    "device": "cuda",
    "num_workers": 2,
    "chkpt": 'model_P_SP2_32_32_32',
}


def test(cfg):
    print("=== Testing ===")
    model = mylayer.CAE_16(2)
    model.load_state_dict(torch.load(f"{cfg.base_dir}/{cfg.exp_name}/chkpt/{cfg.chkpt}"))

    model.eval()
    if cfg.device == "cuda":
        model.cuda()

    print("Loaded model")

    dataset = utils.ImageFolder720p(cfg.dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=cfg.shuffle)

    print("Loaded data")
    psnr = []
    ssim = []

    for batch_idx, data in enumerate(dataloader, start=1):
        img, patches, _ = data
        if cfg.device == 'cuda':
            patches = patches.cuda()

        if batch_idx > 20:
            return np.array(psnr), np.array(ssim)

        out = torch.zeros(6, 10, 3, 128, 128)

        for i in range(6):
            for j in range(10):
                x = (patches[:, :, i, j, :, :]).cuda()
                y = model(x)
                out[i, j] = y.data

        out = np.transpose(out, (0, 3, 1, 4, 2))
        out = np.reshape(out, (768, 1280, 3))

        ssim.append(structural_similarity(np.transpose(img[0].numpy(), (1, 2, 0)), out.numpy(), multichannel=True))

        out = np.transpose(out, (2, 0, 1))

        psnr.append(peak_signal_noise_ratio(img[0].numpy(), out.numpy()))

        y = torch.cat((img[0], out), dim=2)
        utils.save_imgs(imgs=y.unsqueeze(0), to_size=(3, 768, 2 * 1280),
                        name=f"{cfg.base_dir}/{cfg.exp_name}/out/test_{batch_idx}.png")
    return np.array(psnr), np.array(ssim)


if __name__ == '__main__':
    cfg = argparse.Namespace(**cfg)
    test(cfg)
