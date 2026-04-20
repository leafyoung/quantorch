from __future__ import annotations

import json
import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def _extract_batch_tensor(real_data):
    if isinstance(real_data, (tuple, list)):
        return real_data[0]
    return real_data


def _emit_event(payload: dict) -> None:
    LOGGER.info(json.dumps(payload, sort_keys=True))


def train_gan(
    generator,
    discriminator,
    data_loader,
    num_epochs=1000,
    lr=0.0002,
    *,
    log_interval: int = 100,
    progress_callback: Callable[[dict], None] | None = None,
):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_summary = None

        for real_data in data_loader:
            real_batch = _extract_batch_tensor(real_data)
            batch_size = real_batch.size(0)
            device = real_batch.device
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            optimizer_d.zero_grad()
            outputs = discriminator(real_batch)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            noise = torch.randn(batch_size, generator.input_dim, device=device)
            fake_data = generator(noise)
            outputs = discriminator(fake_data.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            outputs = discriminator(fake_data)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

            epoch_summary = {
                "event": "gan_training_progress",
                "module": __name__,
                "function": "train_gan",
                "status": "running",
                "epoch": epoch,
                "num_epochs": num_epochs,
                "discriminator_loss": float((d_loss_real + d_loss_fake).item()),
                "generator_loss": float(g_loss.item()),
            }

        if epoch_summary and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            _emit_event(epoch_summary)
            if progress_callback is not None:
                progress_callback(epoch_summary)
