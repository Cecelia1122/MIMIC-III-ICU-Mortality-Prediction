"""
Training module for MIMIC-III models.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils. data import DataLoader
from torch. optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class Trainer: 
    """Trainer class for MIMIC-III models."""
    
    def __init__(
        self,
        model: nn. Module,
        train_loader:  DataLoader,
        val_loader: DataLoader,
        device:  torch.device,
        model_type: str = 'multimodal',
        learning_rate:  float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch. Tensor] = None
    ):
        self.model = model. to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_type = model_type
        
        if class_weights is not None: 
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_auc':  [],
            'val_loss':  [],
            'val_acc':  [],
            'val_auc': [],
            'learning_rates': []
        }
        
        self.best_val_auc = 0.0
        self.best_model_state = None
    
    def _forward(self, batch):
        """Forward pass based on model type."""
        if self.model_type == 'mlp': 
            static, labels = batch
            static = static.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(static)
        elif self.model_type in ['lstm', 'transformer']: 
            time_series, labels = batch
            time_series = time_series.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(time_series)
        else:  # multimodal
            static, time_series, labels = batch
            static = static.to(self.device)
            time_series = time_series.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(static, time_series)
        
        return outputs, labels
    
    def train_epoch(self) -> Tuple[float, float, float]: 
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            outputs, labels = self._forward(batch)
            
            loss = self. criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils. clip_grad_norm_(self. model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            
            probs = torch. softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels. cpu().numpy())
            all_probs.extend(probs. detach().cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(self. train_loader. dataset)
        epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        
        from sklearn.metrics import roc_auc_score
        try:
            epoch_auc = roc_auc_score(all_labels, all_probs)
        except:
            epoch_auc = 0.5
        
        return epoch_loss, epoch_acc, epoch_auc
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]: 
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            outputs, labels = self._forward(batch)
            
            loss = self.criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs. argmax(dim=1)
            
            all_preds. extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        
        from sklearn. metrics import roc_auc_score
        try:
            epoch_auc = roc_auc_score(all_labels, all_probs)
        except:
            epoch_auc = 0.5
        
        return epoch_loss, epoch_acc, epoch_auc
    
    def train(
        self,
        num_epochs: int = 50,
        save_dir: str = 'checkpoints',
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc, train_auc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_auc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_auc)
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_auc'].append(train_auc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['learning_rates'].append(current_lr)
            
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
            
            # Save best model
            if val_auc > self.best_val_auc:
                self. best_val_auc = val_auc
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, save_dir / 'best_model.pth')
                
                print(f"✓ New best model saved!  Val AUC: {val_auc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save history
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def compute_class_weights(train_loader: DataLoader, model_type: str = 'multimodal') -> torch.Tensor:
    """Compute class weights for imbalanced data."""
    all_labels = []
    
    for batch in train_loader:
        if model_type == 'mlp':
            _, labels = batch
        elif model_type in ['lstm', 'transformer']:
            _, labels = batch
        else: 
            _, _, labels = batch
        all_labels.extend(labels.numpy())
    
    all_labels = np.array(all_labels)
    class_counts = np.bincount(all_labels)
    
    # Inverse frequency weighting
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * len(weights)
    
    return torch. tensor(weights, dtype=torch. float32)