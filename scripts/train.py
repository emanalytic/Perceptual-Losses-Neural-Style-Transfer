from torch.utils.data import DataLoader

from scripts.dataloader import TrainingDataset
from scripts.transformer_network import ImageTransformerNet
from scripts.loss import PerceptualLoss
from scripts.vgg import ForwardHookManager

from utils.utils import *
from utils.config import Config

import torch.optim as optim
import torch.nn as nn
import torch

from tqdm import tqdm

config = Config()

ALPHA = config.get('settings', 'alpha')
BETA = config.get('settings', 'beta')
LR = config.get('settings', 'lr')
CONTENT_DIR = config.get('paths', 'content_dir')
BATCH_SIZE = int(config.get('settings', 'batch_size'))
EPOCHS  = int(config.get('settings', 'epochs'))
STYLE_IMAGE = config.get('paths', 'style_image')


network = ImageTransformerNet()
loss_function = PerceptualLoss(ALPHA, BETA)
optimizer = optim.Adam(network.parameters(), lr=LR)
dataset = TrainingDataset(CONTENT_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
hook_manager = ForwardHookManager()

def train(
        dataloader,
        network,
        vgg,
        loss_function,
        optimizer,
        style_image,
        hook_manager,
        n_epochs,
        DEVICE='cuda',
        BATCH=32,
):
    network.train()
    total_steps = n_epochs * len(dataloader)
    progress_bar = tqdm(total=total_steps, desc="Training Progress", unit="step")

    for epoch in range(1, n_epochs + 1):
        total_loss = []
        for image in dataloader:
            torch.cuda.empty_cache()

            optimizer.zero_grad()
            image = image.to(DEVICE, non_blocking=True)

            y_hat = network(image)
            vgg(y_hat)
            y_hat = hook_manager.layer_outputs
            hook_manager.clear_outputs()

            vgg(image)
            y_content = hook_manager.layer_outputs
            hook_manager.clear_outputs()

            style_target = style_image.to(DEVICE).repeat(BATCH, 1, 1, 1)
            vgg(style_target)
            y_style = hook_manager.layer_outputs
            hook_manager.clear_outputs()

            loss = loss_function(y_hat, y_style[:len(y_hat)], y_content)
            total_loss.append(loss.item())
            loss.backward()

            optimizer.step()

            del y_hat, y_content, y_style, style_target
            torch.cuda.empty_cache()

            avg_loss = sum(total_loss) / len(total_loss)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
            progress_bar.update(1)

    progress_bar.close()
    del hook_manager
    torch.cuda.empty_cache()


def main():
    train(dataloader,
                  network,
                  loss_function,
                  optimizer,
                  style_image,
                  hook_manager,
                  EPOCHS)

    save_model_checkpoint(network, optimizer, 2, "style-transfer-alpha1e5-beta1e10-epoch-2")


if __name__ == "__main__":
    main()
