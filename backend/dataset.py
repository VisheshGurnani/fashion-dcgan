from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fashion_mnist_loader(batch_size: int, data_dir: str | Path | None = None) -> DataLoader:
    """Load FashionMNIST as normalized 28x28 grayscale tensors.

    Keep the dataset inside this project; torchvision downloads it if needed.
    """
    if data_dir is not None:
        root = Path(data_dir).expanduser().resolve()
    else:
        root = PROJECT_ROOT / "data"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.FashionMNIST(root=str(root), train=True, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
