"""
Model architectures for MIMIC-III ICU prediction.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class MLPModel(nn.Module):
    """Simple MLP for static features only."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn. Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, static:  torch.Tensor) -> torch.Tensor:
        return self. network(static)


class LSTMModel(nn.Module):
    """LSTM for time series data."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, time_series: torch.Tensor) -> torch.Tensor:
        # time_series:  (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(time_series)
        
        # Use last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]
        
        return self.classifier(hidden)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0:: 2] = torch.sin(position * div_term)
        pe[:, 0, 1:: 2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class TransformerModel(nn. Module):
    """Transformer for time series data."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers:  int = 2,
        dim_feedforward: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        max_len: int = 100
    ):
        super().__init__()
        
        self. input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, time_series: torch.Tensor) -> torch.Tensor:
        # time_series: (batch, seq_len, features)
        x = self.input_projection(time_series)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return self.classifier(x)


class MultimodalFusionModel(nn.Module):
    """
    Multimodal model combining static features and time series.
    
    Architecture:
        - Static branch: MLP
        - Temporal branch: LSTM or Transformer
        - Fusion:  Concatenation + MLP
    """
    
    def __init__(
        self,
        static_dim: int,
        ts_input_dim: int,
        ts_hidden_dim: int = 128,
        static_hidden_dim: int = 64,
        fusion_hidden_dim: int = 128,
        num_classes: int = 2,
        dropout:  float = 0.3,
        temporal_model:  str = 'lstm'  # 'lstm' or 'transformer'
    ):
        super().__init__()
        
        # Static branch
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, static_hidden_dim * 2),
            nn.BatchNorm1d(static_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(static_hidden_dim * 2, static_hidden_dim),
            nn.ReLU()
        )
        
        # Temporal branch
        self.temporal_model_type = temporal_model
        
        if temporal_model == 'lstm':
            self.temporal_encoder = nn.LSTM(
                input_size=ts_input_dim,
                hidden_size=ts_hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=True
            )
            temporal_output_dim = ts_hidden_dim * 2
        else:  # transformer
            self.temporal_encoder = TransformerModel(
                input_dim=ts_input_dim,
                d_model=ts_hidden_dim,
                num_classes=ts_hidden_dim,  # Output hidden dim
                dropout=dropout
            )
            temporal_output_dim = ts_hidden_dim
        
        # Fusion
        fusion_input_dim = static_hidden_dim + temporal_output_dim
        
        self.fusion = nn.Sequential(
            nn. Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fusion_hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        static: torch.Tensor,
        time_series: torch.Tensor
    ) -> torch.Tensor:
        # Encode static features
        static_features = self.static_encoder(static)
        
        # Encode temporal features
        if self.temporal_model_type == 'lstm': 
            lstm_out, (h_n, c_n) = self.temporal_encoder(time_series)
            temporal_features = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            # For transformer, we need to modify to get features
            x = self.temporal_encoder. input_projection(time_series)
            x = self.temporal_encoder.pos_encoder(x)
            x = self.temporal_encoder.transformer_encoder(x)
            temporal_features = x. mean(dim=1)
        
        # Fusion
        combined = torch.cat([static_features, temporal_features], dim=1)
        return self.fusion(combined)


def get_model(
    model_type: str,
    static_dim: int = None,
    ts_input_dim: int = None,
    num_classes: int = 2,
    device: Optional[torch.device] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'mlp', 'lstm', 'transformer', or 'multimodal'
        static_dim:  Dimension of static features
        ts_input_dim: Dimension of time series features
        num_classes:  Number of output classes
        device: Device to move model to
        
    Returns:
        Initialized model
    """
    if model_type == 'mlp': 
        model = MLPModel(
            input_dim=static_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'lstm':
        model = LSTMModel(
            input_dim=ts_input_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            input_dim=ts_input_dim,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'multimodal':
        model = MultimodalFusionModel(
            static_dim=static_dim,
            ts_input_dim=ts_input_dim,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type:  {model_type}")
    
    if device: 
        model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model:  {model_type} - {total_params: ,} parameters")
    
    return model