"""Dataset factory — unified data loading for all supported datasets."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# (dataset_class, mean, std, input_channels, input_size, num_classes)
DATASETS = {
    "MNIST": (datasets.MNIST, (0.1307,), (0.3081,), 1, 28, 10),
    "FashionMNIST": (datasets.FashionMNIST, (0.2860,), (0.3530,), 1, 28, 10),
    "CIFAR10": (datasets.CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 3, 32, 10),
}


def get_info(name):
    """Return (input_channels, input_size, num_classes) for a dataset."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
    _, _, _, ch, sz, nc = DATASETS[name]
    return ch, sz, nc


def get_loaders(name, batch_size_train=128, batch_size_test=256, data_dir="./data"):
    """Return (train_loader, test_loader) for the named dataset."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")

    ds_cls, mean, std, _ch, _sz, _nc = DATASETS[name]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = ds_cls(data_dir, train=True, download=True, transform=transform)
    test_ds = ds_cls(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size_test, shuffle=False, num_workers=0)

    return train_loader, test_loader
