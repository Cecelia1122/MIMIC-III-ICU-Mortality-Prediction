"""
Debug script for MIMIC-III project.

Usage:
    python debug.py --step 1    # Test data extraction
    python debug.py --step 2    # Test preprocessing
    python debug.py --step 3    # Test dataset loading
    python debug.py --step 4    # Test models
    python debug.py --step 5    # Test training
    python debug.py --step all  # Test all
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_success(text):
    print(f"  ✓ {text}")


def print_error(text):
    print(f"  ✗ {text}")


def print_info(text):
    print(f"  ℹ {text}")


def test_imports():
    """Test all imports."""
    print_header("STEP 0: Testing Imports")
    
    try:
        from src.utils import set_seed, get_device
        print_success("utils imports OK")
        
        from src.data_extraction import MIMICDataExtractor
        print_success("data_extraction imports OK")
        
        from src.preprocessing import MIMICPreprocessor
        print_success("preprocessing imports OK")
        
        from src.dataset import MIMICDataset, get_data_loaders
        print_success("dataset imports OK")
        
        from src.models import get_model, MLPModel, LSTMModel, TransformerModel, MultimodalFusionModel
        print_success("models imports OK")
        
        from src.train import Trainer, compute_class_weights
        print_success("train imports OK")
        
        from src.evaluate import Evaluator
        print_success("evaluate imports OK")
        
        return True
    except ImportError as e:
        print_error(f"Import failed: {e}")
        return False


def test_data_extraction():
    """Test data extraction (requires MIMIC-III data)."""
    print_header("STEP 1: Testing Data Extraction")
    
    mimic_paths = [
        Path('./data/mimic-iii-clinical-database-1.4'),
        Path('./data/mimic-iii'),
        Path('./data/physionet.org/files/mimiciii/1.4'),
    ]
    
    mimic_path = None
    for p in mimic_paths:
        if p.exists():
            mimic_path = p
            break
    
    if mimic_path is None:
        print_info("MIMIC-III data not found in standard locations")
        print_info("Skipping data extraction test")
        return True
    
    try:
        from src.data_extraction import MIMICDataExtractor
        
        extractor = MIMICDataExtractor(str(mimic_path))
        print_success(f"Created extractor for {mimic_path}")
        
        # Test cohort extraction (limited for speed)
        print_info("Extracting small cohort sample...")
        cohort = extractor.extract_cohort(min_age=18, min_los_hours=48)
        print_success(f"Extracted cohort: {len(cohort)} patients")
        
        return True
    except Exception as e: 
        print_error(f"Data extraction failed: {e}")
        return False


def test_preprocessing():
    """Test preprocessing functions."""
    print_header("STEP 2: Testing Preprocessing")
    
    try:
        from src.preprocessing import MIMICPreprocessor
        
        preprocessor = MIMICPreprocessor()
        print_success("Created preprocessor")
        
        # Test aggregation
        import pandas as pd
        test_data = pd.DataFrame({
            'HOURS': [1, 2, 3, 4, 5],
            'VITAL_NAME': ['heart_rate'] * 5,
            'VALUENUM': [70, 75, 80, 72, 78]
        })
        
        features = preprocessor._aggregate_time_series(
            test_data, 'VALUENUM', 'VITAL_NAME', ['heart_rate', 'sbp']
        )
        print_success(f"Aggregation test: heart_rate_mean = {features['heart_rate_mean']:.2f}")
        
        # Test time series creation
        ts = preprocessor._create_time_series(
            test_data, 'VALUENUM', 'VITAL_NAME', ['heart_rate'],
            hours=48, interval=4
        )
        print_success(f"Time series shape: {ts.shape}")
        
        return True
    except Exception as e:
        print_error(f"Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset loading."""
    print_header("STEP 3: Testing Dataset")
    
    data_dir = Path('./data/processed')
    
    if not (data_dir / 'train. pkl').exists():
        print_info("Processed data not found")
        print_info("Creating dummy data for testing...")
        
        # Create dummy data
        import pickle
        
        data_dir.mkdir(parents=True, exist_ok=True)
        
        n_samples = 100
        static_dim = 68
        ts_len = 12
        ts_features = 20
        
        for split in ['train', 'val', 'test']:
            data = {
                'static':  np.random.randn(n_samples, static_dim).astype(np.float32),
                'time_series': np.random.randn(n_samples, ts_len, ts_features).astype(np.float32),
                'labels': np.random.randint(0, 2, n_samples).astype(np.int64),
                'icustay_ids': np.arange(n_samples)
            }
            with open(data_dir / f'{split}.pkl', 'wb') as f:
                pickle.dump(data, f)
        
        # Create feature info
        import json
        feature_info = {
            'num_static_features':  static_dim,
            'num_timesteps': ts_len,
            'num_ts_features': ts_features
        }
        with open(data_dir / 'feature_info.json', 'w') as f:
            json.dump(feature_info, f)
        
        print_success("Created dummy data")
    
    try:
        from src.dataset import MIMICDataset, get_data_loaders, load_feature_info
        
        # Test dataset
        train_dataset = MIMICDataset(data_dir / 'train.pkl')
        print_success(f"Loaded train dataset:  {len(train_dataset)} samples")
        
        # Test single item
        static, ts, label = train_dataset[0]
        print_success(f"Sample:  static={static.shape}, ts={ts.shape}, label={label}")
        
        # Test data loaders
        train_loader, val_loader, test_loader = get_data_loaders(
            str(data_dir), batch_size=16
        )
        batch = next(iter(train_loader))
        print_success(f"Batch:  static={batch[0].shape}, ts={batch[1].shape}")
        
        return True
    except Exception as e:
        print_error(f"Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """Test model architectures."""
    print_header("STEP 4: Testing Models")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_info(f"Using device: {device}")
    
    static_dim = 68
    ts_input_dim = 20
    batch_size = 4
    seq_len = 12
    
    try:
        from src.models import get_model
        
        # Test MLP
        model = get_model('mlp', static_dim=static_dim, device=device)
        x = torch.randn(batch_size, static_dim).to(device)
        out = model(x)
        assert out.shape == (batch_size, 2)
        print_success(f"MLP:  input {x.shape} -> output {out.shape}")
        del model
        
        # Test LSTM
        model = get_model('lstm', ts_input_dim=ts_input_dim, device=device)
        x = torch.randn(batch_size, seq_len, ts_input_dim).to(device)
        out = model(x)
        assert out.shape == (batch_size, 2)
        print_success(f"LSTM: input {x.shape} -> output {out.shape}")
        del model
        
        # Test Transformer
        model = get_model('transformer', ts_input_dim=ts_input_dim, device=device)
        x = torch.randn(batch_size, seq_len, ts_input_dim).to(device)
        out = model(x)
        assert out.shape == (batch_size, 2)
        print_success(f"Transformer: input {x.shape} -> output {out.shape}")
        del model
        
        # Test Multimodal
        model = get_model('multimodal', static_dim=static_dim, ts_input_dim=ts_input_dim, device=device)
        static = torch.randn(batch_size, static_dim).to(device)
        ts = torch.randn(batch_size, seq_len, ts_input_dim).to(device)
        out = model(static, ts)
        assert out.shape == (batch_size, 2)
        print_success(f"Multimodal: inputs ({static.shape}, {ts.shape}) -> output {out.shape}")
        del model
        
        return True
    except Exception as e:
        print_error(f"Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training():
    """Test training loop."""
    print_header("STEP 5: Testing Training")
    
    device = torch.device('cuda' if torch.cuda. is_available() else 'cpu')
    
    try:
        from src. models import get_model
        from src.train import Trainer, compute_class_weights
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy data
        static_dim = 68
        ts_input_dim = 20
        seq_len = 12
        n_samples = 64
        
        static = torch.randn(n_samples, static_dim)
        ts = torch.randn(n_samples, seq_len, ts_input_dim)
        labels = torch.randint(0, 2, (n_samples,))
        
        dataset = TensorDataset(static, ts, labels)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        print_success("Created dummy data loaders")
        
        # Create model
        model = get_model(
            'multimodal',
            static_dim=static_dim,
            ts_input_dim=ts_input_dim,
            device=device
        )
        print_success("Created model")
        
        # Compute class weights
        class_weights = compute_class_weights(train_loader, model_type='multimodal')
        print_success(f"Class weights: {class_weights}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            model_type='multimodal',
            learning_rate=1e-3,
            class_weights=class_weights
        )
        print_success("Created trainer")
        
        # Run one epoch
        print_info("Running 1 training epoch...")
        train_loss, train_acc, train_auc = trainer.train_epoch()
        print_success(f"Train:  loss={train_loss:.4f}, acc={train_acc:.4f}, auc={train_auc:.4f}")
        
        val_loss, val_acc, val_auc = trainer.validate()
        print_success(f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}, auc={val_auc:.4f}")
        
        return True
    except Exception as e:
        print_error(f"Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Debug script')
    parser.add_argument('--step', type=str, default='all', help='Step to test:  0-5 or "all"')
    args = parser.parse_args()
    
    print("=" * 60)
    print(" MIMIC-III ICU PREDICTION - DEBUG SCRIPT")
    print("=" * 60)
    
    steps = {
        '0': ('Imports', test_imports),
        '1': ('Data Extraction', test_data_extraction),
        '2': ('Preprocessing', test_preprocessing),
        '3': ('Dataset', test_dataset),
        '4': ('Models', test_models),
        '5': ('Training', test_training),
    }
    
    if args. step == 'all':
        steps_to_run = list(steps.keys())
    else:
        steps_to_run = [args.step]
    
    results = {}
    for step_num in steps_to_run:
        if step_num in steps:
            name, func = steps[step_num]
            try:
                results[step_num] = func()
            except Exception as e:
                print_error(f"Step {step_num} crashed: {e}")
                results[step_num] = False
    
    # Summary
    print_header("SUMMARY")
    all_passed = True
    for step_num, passed in results.items():
        name = steps[step_num][0]
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  Step {step_num} ({name}): {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n  🎉 All tests passed!")
    else:
        print("\n  ⚠️ Some tests failed.")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())