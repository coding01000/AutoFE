import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import utils
from others import mylayer

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


def train(cfg) -> None:
    print("=== Training ===")

    utils.mkdir(cfg)
    model = mylayer.CAE_16(2)
    model.train()
    if cfg.device == "cuda":
        model.cuda()
    print(f"Model loaded on {cfg.device}")

    dataset = utils.ImageFolder720p(cfg.dataset_path)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers)
    print("Data loaded")

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    loss_criterion = nn.MSELoss()
    # scheduler = ...

    avg_loss, epoch_avg = 0.0, 0.0
    ts = 0

    # train-loop
    for epoch_idx in range(cfg.start_epoch, cfg.num_epochs + 1):

        print(epoch_idx, ' is training')
        for batch_idx, data in enumerate(dataloader, start=1):
            img, patches, _ = data

            if cfg.device == "cuda":
                patches = patches.cuda()

            avg_loss_per_image = 0.0

            for i in range(6):
                for j in range(10):
                    optimizer.zero_grad()

                    x = patches[:, :, i, j, :, :]
                    y = model(x)
                    loss = loss_criterion(y, x)

                    avg_loss_per_image += (1 / 60) * loss.item()

                    loss.backward()
                    optimizer.step()

            avg_loss += avg_loss_per_image
            epoch_avg += avg_loss_per_image

            if batch_idx % cfg.save_every == 0:
                out = torch.zeros(6, 10, 3, 128, 128)
                for i in range(6):
                    for j in range(10):
                        x = (patches[0, :, i, j, :, :].unsqueeze(0)).cuda()
                        out[i, j] = model(x).cpu().data

                out = np.transpose(out, (0, 3, 1, 4, 2))
                out = np.reshape(out, (768, 1280, 3))
                out = np.transpose(out, (2, 0, 1))

                y = torch.cat((img[0], out), dim=2).unsqueeze(0)
                utils.save_imgs(imgs=y, to_size=(3, 768, 2 * 1280),
                                name=f"{cfg.base_dir}/{cfg.exp_name}/out3/out_{epoch_idx}_{batch_idx}.png")

    torch.save(model.state_dict(), f"{cfg.base_dir}/{cfg.exp_name}/chkpt/{cfg.chkpt}")


if __name__ == '__main__':
    cfg = argparse.Namespace(**cfg)
    train(cfg)
