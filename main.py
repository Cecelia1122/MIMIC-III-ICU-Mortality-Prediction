"""
Main script for MIMIC-III ICU Mortality Prediction. 

Usage:
    # Step 1: Extract and preprocess data
    python main.py --mode preprocess --mimic_path ./data/mimic-iii-clinical-database-1.4
    
    # Step 2: Train model
    python main.py --mode train --model multimodal --epochs 50
    
    # Step 3: Evaluate
    python main.py --mode evaluate --model multimodal
    
    # Full pipeline
    python main.py --mode full --mimic_path ./data/mimic-iii-clinical-database-1.4

Models: 
    - mlp:          MLP on static features only
    - lstm:        LSTM on time series only
    - transformer: Transformer on time series only  
    - multimodal:   Fusion of static + time series (RECOMMENDED)
"""

import argparse
from pathlib import Path
import json
import shutil

import torch
from src.utils import set_seed, get_device, print_header
from src.data_extraction import MIMICDataExtractor
from src.preprocessing import MIMICPreprocessor
from src.dataset import get_data_loaders, load_feature_info
from src.models import get_model
from src.train import Trainer, compute_class_weights
from src.evaluate import Evaluator
import warnings


warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(
        description='MIMIC-III ICU Mortality Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode
    parser.add_argument(
        '--mode', type=str,
        choices=['preprocess', 'train', 'evaluate', 'full'],
        default='train',
        help='Mode to run'
    )
    
    # Data paths
    parser.add_argument(
        '--mimic_path', type=str,
        default='./data/mimic-iii-clinical-database-1.4',
        help='Path to MIMIC-III data'
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='./data/processed',
        help='Path to processed data'
    )
    
    # Preprocessing
    parser.add_argument('--hours', type=int, default=48, help='Hours of data to extract')
    parser.add_argument('--interval', type=int, default=4, help='Time series interval (hours)')
    
    # Model
    parser.add_argument(
        '--model', type=str,
        choices=['mlp', 'lstm', 'transformer', 'multimodal'],
        default='multimodal',
        help='Model type'
    )
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def preprocess(args):
    """Extract and preprocess MIMIC-III data."""
    print_header("DATA EXTRACTION & PREPROCESSING")
    
    # Check MIMIC path
    mimic_path = Path(args.mimic_path)
    if not mimic_path.exists():
        print(f"✗ MIMIC-III path not found: {mimic_path}")
        print("\nPlease ensure MIMIC-III data is extracted to the specified path.")
        print("Expected files:  PATIENTS.csv. gz, ADMISSIONS.csv.gz, ICUSTAYS.csv.gz, etc.")
        return False
    
    print(f"MIMIC-III path: {mimic_path}")
    
    # Extract data
    extractor = MIMICDataExtractor(str(mimic_path))
    cohort, vitals, labs, diagnoses = extractor.extract_all(
        hours=args.hours,
        min_age=18,
        min_los_hours=24,
        max_los_days=30
    )
    
    # Preprocess
    preprocessor = MIMICPreprocessor()
    feature_info = preprocessor.prepare_dataset(
        cohort=cohort,
        vitals_dict=vitals,
        labs_dict=labs,
        diagnoses_dict=diagnoses,
        output_dir=args.data_dir,
        hours=args.hours,
        interval=args.interval
    )
    
    print("\n✓ Preprocessing complete!")
    print(f"  Data saved to: {args.data_dir}/")
    
    return True


def train(args, device):
    """Train model."""
    print_header("TRAINING")
    
    # Check processed data
    data_dir = Path(args.data_dir)
    if not (data_dir / 'train.pkl').exists():
        print(f"✗ Processed data not found in {data_dir}")
        print("Please run preprocessing first:  python main.py --mode preprocess")
        return None, None
    
    # Load feature info
    feature_info = load_feature_info(args.data_dir)
    print(f"Feature info:")
    print(f"  Static features: {feature_info['num_static_features']}")
    print(f"  Time steps: {feature_info['num_timesteps']}")
    print(f"  Time series features: {feature_info['num_ts_features']}")
    
    # Determine what data to use based on model
    if args.model == 'mlp':
        use_static, use_ts = True, False
    elif args.model in ['lstm', 'transformer']: 
        use_static, use_ts = False, True
    else:   # multimodal
        use_static, use_ts = True, True
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        use_static=use_static,
        use_time_series=use_ts
    )
    
    # Compute class weights
    class_weights = compute_class_weights(train_loader, model_type=args.model)
    print(f"Class weights: {class_weights}")
    
    # Create model
    model = get_model(
        model_type=args.model,
        static_dim=feature_info['num_static_features'],
        ts_input_dim=feature_info['num_ts_features'],
        num_classes=2,
        device=device
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_type=args.model,
        learning_rate=args.lr,
        class_weights=class_weights
    )
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        early_stopping_patience=10
    )
    
    # Save with model-specific name
    checkpoint_name = f'best_model_{args.model}.pth'
    src = Path(args.save_dir) / 'best_model. pth'
    dst = Path(args.save_dir) / checkpoint_name
    if src.exists():
        shutil.copy(src, dst)
        print(f"✓ Model saved to:  {dst}")
    
    # Plot training history
    results_dir = Path(args.results_dir) / 'figures'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = Evaluator(model, val_loader, device, model_type=args.model)
    evaluator.plot_training_history(
        history,
        str(results_dir / f'training_history_{args.model}.png')
    )
    
    return model, history


def evaluate(args, device, model=None):
    """Evaluate model."""
    print_header("EVALUATION")
    
    # Load feature info
    feature_info = load_feature_info(args.data_dir)
    
    # Determine what data to use
    if args.model == 'mlp':
        use_static, use_ts = True, False
    elif args.model in ['lstm', 'transformer']:
        use_static, use_ts = False, True
    else: 
        use_static, use_ts = True, True
    
    # Get test loader
    _, _, test_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        use_static=use_static,
        use_time_series=use_ts
    )
    
    # Load model if not provided
    if model is None: 
        model = get_model(
            model_type=args. model,
            static_dim=feature_info['num_static_features'],
            ts_input_dim=feature_info['num_ts_features'],
            num_classes=2,
            device=device
        )
        
        # Find checkpoint
        checkpoint_paths = [
            Path(args.checkpoint) if args.checkpoint else None,
            Path(args.save_dir) / f'best_model_{args.model}.pth',
            Path(args.save_dir) / 'best_model.pth',
        ]
        
        loaded = False
        for cp in checkpoint_paths:
            if cp and cp.exists():
                checkpoint = torch.load(cp, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded checkpoint:  {cp}")
                if 'val_auc' in checkpoint:
                    print(f"  Validation AUC: {checkpoint['val_auc']:.4f}")
                loaded = True
                break
        
        if not loaded: 
            print("✗ No checkpoint found!")
            return None
    
    # Evaluate
    results_dir = Path(args.results_dir) / 'figures'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = Evaluator(model, test_loader, device, model_type=args.model)
    metrics = evaluator.full_evaluation(str(results_dir))
    
    return metrics


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    print("=" * 60)
    print(" MIMIC-III ICU MORTALITY PREDICTION")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    
    # Create directories
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    model = None
    
    # ==================== PREPROCESS ====================
    if args.mode in ['preprocess', 'full']: 
        success = preprocess(args)
        if not success and args.mode == 'preprocess':
            return
    
    # ==================== TRAIN ====================
    if args.mode in ['train', 'full']: 
        model, history = train(args, device)
        if model is None and args.mode == 'train': 
            return
    
    # ==================== EVALUATE ====================
    if args.mode in ['evaluate', 'full']:
        evaluate(args, device, model)
    
    # ==================== COMPLETE ====================
    print_header("COMPLETE!")
    print(f"Checkpoints:  {args.save_dir}/")
    print(f"Results: {args.results_dir}/")


if __name__ == '__main__':
    main()