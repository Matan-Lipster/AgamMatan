import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, dropout=0.1, max_seq_len=5000, d_model=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe = torch.zeros(max_seq_len, d_model)  # [max_seq_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # [batch, seq, d_model]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 input_size: int,
                 max_seq_len: int = 100,
                 dim_val: int = 256,
                 n_heads: int = 4,
                 n_encoder_layers: int = 3,
                 dropout: float = 0.1,
                 dim_feedforward: int = 512,
                 num_predicted_features: int = 1):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_size)
        self.input_layer = nn.Linear(input_size, dim_val)
        self.pos_encoder = PositionalEncoder(d_model=dim_val, max_seq_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Output layer: maps from encoded vector to predicted values
        self.output_layer = nn.Linear(dim_val, num_predicted_features)

    def forward(self, src):
        """
        src shape: [batch, seq_len, input_size]
        output: [batch, num_predicted_features]
        """
        src = self.input_norm(src)
        src = self.input_layer(src)  # [batch, seq, dim_val]
        src = self.pos_encoder(src)
        encoded = self.encoder(src)  # [batch, seq, dim_val]

        # Global representation: average over time (TRs)
        pooled = torch.mean(encoded, dim=1)  # [batch, dim_val]
        output = self.output_layer(pooled)   # [batch, num_predicted_features]
        return output