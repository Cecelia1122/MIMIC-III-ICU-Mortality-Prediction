"""
Evaluation module for MIMIC-III models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score
)
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Evaluator:
    """Evaluator class for MIMIC-III models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        model_type: str = 'multimodal'
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.model_type = model_type
        
        self.predictions = None
        self.probabilities = None
        self.true_labels = None
    
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
    
    @torch.no_grad()
    def run_inference(self):
        """Run inference on test set."""
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        for batch in tqdm(self.test_loader, desc='Evaluating'):
            outputs, labels = self._forward(batch)
            
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        self.predictions = np.array(all_preds)
        self.probabilities = np.array(all_probs)
        self.true_labels = np.array(all_labels)
    
    def get_metrics(self) -> Dict:
        """Calculate all metrics."""
        if self.predictions is None:
            self.run_inference()
        
        # Basic metrics
        accuracy = accuracy_score(self.true_labels, self.predictions)
        f1 = f1_score(self.true_labels, self. predictions)
        
        # AUC-ROC
        try:
            auroc = auc(*roc_curve(self.true_labels, self.probabilities[: , 1])[: 2])
        except:
            auroc = 0.5
        
        # AUC-PR
        try: 
            auprc = average_precision_score(self.true_labels, self.probabilities[: , 1])
        except:
            auprc = 0.0
        
        return {
            'accuracy': accuracy,
            'f1_score':  f1,
            'auroc': auroc,
            'auprc': auprc
        }
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix."""
        if self.predictions is None:
            self.run_inference()
        
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Survived', 'Deceased'],
            yticklabels=['Survived', 'Deceased'],
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix - ICU Mortality Prediction')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curve."""
        if self. probabilities is None:
            self. run_inference()
        
        fpr, tpr, _ = roc_curve(self.true_labels, self.probabilities[: , 1])
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - ICU Mortality Prediction')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, save_path:  Optional[str] = None) -> plt.Figure:
        """Plot Precision-Recall curve."""
        if self.probabilities is None:
            self.run_inference()
        
        precision, recall, _ = precision_recall_curve(
            self.true_labels, self.probabilities[:, 1]
        )
        ap = average_precision_score(self.true_labels, self.probabilities[:, 1])
        
        fig, ax = plt. subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='#ff7f0e', lw=2, label=f'PR (AP = {ap:.3f})')
        ax.axhline(y=self.true_labels.mean(), color='k', linestyle='--', lw=2, label='Baseline')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve - ICU Mortality Prediction')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot training history."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # AUC
        axes[2].plot(epochs, history['train_auc'], 'b-', label='Train')
        axes[2].plot(epochs, history['val_auc'], 'r-', label='Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].set_title('AUC-ROC')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def full_evaluation(self, save_dir: str = 'results/figures') -> Dict:
        """Run full evaluation and generate plots."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Run inference
        self.run_inference()
        
        # Get metrics
        metrics = self.get_metrics()
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Accuracy:   {metrics['accuracy']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"AUC-ROC:   {metrics['auroc']:.4f}")
        print(f"AUC-PR:    {metrics['auprc']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(
            self.true_labels,
            self.predictions,
            target_names=['Survived', 'Deceased'],
            digits=4
        ))
        
        # Generate plots
        self.plot_confusion_matrix(str(save_dir / 'confusion_matrix.png'))
        self.plot_roc_curve(str(save_dir / 'roc_curve.png'))
        self.plot_precision_recall_curve(str(save_dir / 'pr_curve.png'))
        
        plt.close('all')
        
        print(f"\n✓ Plots saved to {save_dir}/")
        
        return metrics