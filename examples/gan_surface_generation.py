import torch

from torchquantlib.models.stochastic_volatility.gans import Discriminator, Generator, train_gan


def print_progress(event):
    print(event)


input_dim = 100
output_dim = 10
batch_size = 32

generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)
data_loader = torch.utils.data.DataLoader(
    torch.randn(1000, output_dim),
    batch_size=batch_size,
    shuffle=True,
)

train_gan(
    generator,
    discriminator,
    data_loader,
    num_epochs=10,
    log_interval=1,
    progress_callback=print_progress,
)
