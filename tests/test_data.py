"""Tests for autolab.data — dataset factory, info lookup."""

import pytest
from torch.utils.data import DataLoader

from autolab.data import get_loaders, get_info, DATASETS


class TestGetInfo:
    """Test get_info returns correct metadata for each dataset."""

    @pytest.mark.parametrize("name", list(DATASETS.keys()))
    def test_returns_correct_tuple(self, name):
        ch, sz, nc = get_info(name)
        _, _, _, expected_ch, expected_sz, expected_nc = DATASETS[name]
        assert ch == expected_ch
        assert sz == expected_sz
        assert nc == expected_nc

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_info("NoSuchDataset_XYZ")


class TestGetLoaders:
    """Test get_loaders returns proper DataLoader objects."""

    def test_mnist_returns_two_dataloaders(self, tmp_path):
        train_loader, test_loader = get_loaders(
            "MNIST", batch_size_train=32, batch_size_test=32, data_dir=str(tmp_path / "data")
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_unknown_dataset_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_loaders("NoSuchDataset_XYZ", data_dir=str(tmp_path / "data"))
