"""
PyTorch Dataset classes for MIMIC-III data.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader


class MIMICDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-III ICU prediction.
    
    Supports: 
        - Static features only (MLP)
        - Time series only (LSTM/Transformer)
        - Both (Multimodal)
    """
    
    def __init__(
        self,
        data_path: str,
        use_static: bool = True,
        use_time_series: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to . pkl file (train. pkl, val.pkl, or test.pkl)
            use_static: Whether to return static features
            use_time_series: Whether to return time series
        """
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.static = torch.tensor(data['static'], dtype=torch.float32)
        self.time_series = torch.tensor(data['time_series'], dtype=torch.float32)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)
        self.icustay_ids = data['icustay_ids']
        
        self.use_static = use_static
        self.use_time_series = use_time_series
        
        print(f"  Loaded {len(self)} samples")
        print(f"    Static shape: {self.static.shape}")
        print(f"    Time series shape: {self.time_series.shape}")
        print(f"    Mortality rate: {self.labels. float().mean():.2%}")
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ... ]:
        label = self.labels[idx]
        
        if self.use_static and self.use_time_series:
            return self.static[idx], self. time_series[idx], label
        elif self.use_static:
            return self.static[idx], label
        elif self.use_time_series:
            return self.time_series[idx], label
        else: 
            raise ValueError("Must use at least one of static or time_series")
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        unique, counts = np.unique(self. labels.numpy(), return_counts=True)
        return {f"class_{int(u)}": int(c) for u, c in zip(unique, counts)}


def get_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    use_static: bool = True,
    use_time_series: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Directory containing train.pkl, val.pkl, test.pkl
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_static: Whether to use static features
        use_time_series: Whether to use time series
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    
    print("\nLoading datasets...")
    
    print("  Train:")
    train_dataset = MIMICDataset(
        data_dir / 'train.pkl',
        use_static=use_static,
        use_time_series=use_time_series
    )
    
    print("  Validation:")
    val_dataset = MIMICDataset(
        data_dir / 'val.pkl',
        use_static=use_static,
        use_time_series=use_time_series
    )
    
    print("  Test:")
    test_dataset = MIMICDataset(
        data_dir / 'test.pkl',
        use_static=use_static,
        use_time_series=use_time_series
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def load_feature_info(data_dir: str) -> Dict:
    """Load feature information."""
    import json
    with open(Path(data_dir) / 'feature_info.json', 'r') as f:
        return json.load(f)