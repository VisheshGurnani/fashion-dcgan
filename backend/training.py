import csv
import random
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torchvision.utils import save_image

from .dataset import PROJECT_ROOT, fashion_mnist_loader
from .models import LATENT_DIM, Discriminator, Generator


def select_device(preference: str = "Auto") -> torch.device:
    if preference == "CPU":
        return torch.device("cpu")
    if preference == "CUDA" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but this PyTorch installation cannot access a GPU.")
    if preference == "CUDA" or torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _append_losses(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "generator_loss", "discriminator_loss"])
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def train_dcgan(
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.0002,
    device_preference: str = "Auto",
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    seed: int = 42,
) -> dict:
    """Train a DCGAN; callbacks receive epoch metrics and the latest sample grid."""
    if epochs < 1 or batch_size < 2:
        raise ValueError("Epochs must be positive and batch size must be at least 2.")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = select_device(device_preference)
    output_root = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs"
    sample_dir = output_root / "samples"
    checkpoint_dir = output_root / "checkpoints"
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    loader = fashion_mnist_loader(batch_size=batch_size, data_dir=data_dir)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    fixed_noise = torch.randn(25, LATENT_DIM, device=device)
    loss_rows: list[dict] = []
    starting_epoch = 1

    latest_checkpoint = checkpoint_dir / "latest.pt"
    if latest_checkpoint.exists():
        state = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        optimizer_g.load_state_dict(state["optimizer_g"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        starting_epoch = int(state.get("epoch", 0)) + 1

    for epoch in range(starting_epoch, starting_epoch + epochs):
        generator.train()
        discriminator.train()
        total_g = 0.0
        total_d = 0.0
        batches = 0

        for real_images, _ in loader:
            real_images = real_images.to(device)
            batch = real_images.size(0)
            real_targets = torch.ones(batch, device=device)
            fake_targets = torch.zeros(batch, device=device)

            optimizer_d.zero_grad(set_to_none=True)
            real_loss = criterion(discriminator(real_images), real_targets)
            noise = torch.randn(batch, LATENT_DIM, device=device)
            fake_images = generator(noise)
            fake_loss = criterion(discriminator(fake_images.detach()), fake_targets)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            generated = generator(noise)
            g_loss = criterion(discriminator(generated), real_targets)
            g_loss.backward()
            optimizer_g.step()

            total_g += float(g_loss.detach().item())
            total_d += float(d_loss.detach().item())
            batches += 1

        if not batches:
            raise RuntimeError("The dataset did not produce any complete training batches.")
        row = {
            "epoch": epoch,
            "generator_loss": total_g / batches,
            "discriminator_loss": total_d / batches,
        }
        loss_rows.append(row)
        _append_losses(output_root / "losses.csv", [row])

        generator.eval()
        with torch.no_grad():
            preview = generator(fixed_noise).cpu()
        sample_path = sample_dir / f"epoch_{epoch:03d}.png"
        save_image(preview, sample_path, nrow=5, normalize=True, value_range=(-1, 1), padding=2)

        torch.save(
            {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "latent_dim": LATENT_DIM,
                "image_size": 28,
                "losses": loss_rows,
            },
            latest_checkpoint,
        )
        torch.save(generator.state_dict(), checkpoint_dir / f"generator_epoch_{epoch:03d}.pt")

        if progress_callback:
            progress_callback({**row, "sample_path": sample_path, "device": str(device)})

    return {
        "generator": generator,
        "discriminator": discriminator,
        "losses": loss_rows,
        "checkpoint_path": latest_checkpoint,
        "device": str(device),
    }


def generate_images(
    checkpoint_path: str | Path | None = None,
    count: int = 16,
    device_preference: str = "Auto",
) -> torch.Tensor:
    """Load the saved generator and return generated images in [-1, 1]."""
    if not 1 <= count <= 64:
        raise ValueError("Image count must be between 1 and 64.")
    path = Path(checkpoint_path) if checkpoint_path else PROJECT_ROOT / "outputs" / "checkpoints" / "latest.pt"
    if not path.exists():
        raise FileNotFoundError("Train the model first; no checkpoint was found.")
    device = select_device(device_preference)
    state = torch.load(path, map_location=device, weights_only=False)
    generator = Generator(latent_dim=int(state.get("latent_dim", LATENT_DIM))).to(device)
    generator.load_state_dict(state["generator"])
    generator.eval()
    with torch.no_grad():
        return generator(torch.randn(count, LATENT_DIM, device=device)).cpu()
