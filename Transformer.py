import torch
import torchmetrics
import wandb
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix
from torchmetrics import ConfusionMatrix
import random
import math

seq_len = 300 # set to 1 if averaging TRs, else - seq_len = len(TRs)
embedding_dim = 256

# Positionwise Feed-Forward Network
# This network is used within the Transformer model to process the output of the attention layer.
# It applies two linear transformations with a ReLU activation in between and a dropout for regularization.
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)  # First linear layer
        self.linear2 = nn.Linear(hidden, d_model)  # Second linear layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.dropout = nn.Dropout(p=drop_prob)  # Dropout layer for regularization

    def forward(self, x):
        x = self.linear1(x)  # Apply first linear layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.linear2(x)  # Apply second linear layer
        return x

# Positional Encoding
# This class implements the positional encoding to provide information about the position of elements in the sequence.
# It uses sine and cosine functions to encode the positions and adds dropout for regularization.
class PositionalEncoding(nn.Module):
    def __init__(self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512, batch_first: bool = False):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0
        position = torch.arange(max_seq_len).unsqueeze(1)  # Position tensor
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # Divisor term
        pe = torch.zeros( max_seq_len, d_model)  # Positional encoding tensor

        pe[ :, 0::2] = torch.sin(position * div_term)  # Apply sine function
        pe[ :, 1::2] = torch.cos(position * div_term)  # Apply cosine function

        self.register_buffer('pe', pe)  # Register positional encoding tensor

    def forward(self, x):

        # Ensure `pe` is sliced correctly to match `x`
        x = x + self.pe[:x.size(1), :]  # Slice positional encoding correctly
        x = self.dropout(x)
        return x


# Attention Block
# This class implements the attention mechanism used in the Transformer model.
# It includes multi-head attention, dropout, and layer normalization. It also applies a feed-forward network.
class AttentionBlock(nn.Module):
    def __init__(self, num_heads=1, head_size=128, ff_dim=None, dropout=0):
        super(AttentionBlock, self).__init__()

        if ff_dim is None:
            ff_dim = head_size

        self.attention = nn.MultiheadAttention(embed_dim=head_size, num_heads=num_heads, dropout=dropout)  # Multi-head attention
        self.attention_dropout = nn.Dropout(dropout)  # Dropout for attention layer
        self.attention_norm = nn.LayerNorm(head_size, eps=1e-6)  # Layer normalization after attention

        self.ffn = PositionwiseFeedForward(d_model=head_size, hidden=128, drop_prob=0.1)  # Feed-forward network
        self.ff_dropout = nn.Dropout(dropout)  # Dropout for feed-forward network
        self.ff_norm = nn.LayerNorm(ff_dim, eps=1e-6)  # Layer normalization after feed-forward network

    def forward(self, inputs):
        x, attention_scores = self.attention(inputs, inputs, inputs)  # Apply multi-head attention
        x = self.attention_dropout(x)  # Apply dropout
        x = self.attention_norm(inputs + x)  # Apply layer normalization

        x = self.ffn(x)  # Apply feed-forward network
        x = self.ff_dropout(x)  # Apply dropout
        x = self.ff_norm(inputs + x)  # Apply layer normalization
        return x, attention_scores

# Transformer Encoder
# This class implements the encoder part of the Transformer model.
# It includes an input linear layer, positional encoding, multiple attention blocks, and final linear layers for classification.
class TransformerEncoder(nn.Module):
    def __init__(self, num_voxels, classes, time2vec_dim, num_heads=1, head_size=128, ff_dim=None, num_layers=1,
                 dropout=0):
        super(TransformerEncoder, self).__init__()
        self.encoder_input_layer = nn.Linear(in_features=num_voxels, out_features=ff_dim)  # Input linear layer
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.classes = classes
        self.positional_encoding = PositionalEncoding(dropout=0.1, max_seq_len=seq_len, d_model=embedding_dim, batch_first=False)  # Positional encoding
        self.attention_layers = nn.ModuleList(
            [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)])  # Attention blocks
        self.norm = nn.LayerNorm(ff_dim)  # Layer normalization
        self.final_layers = nn.Sequential(
            #nn.Flatten(),  # Flatten the input
            #nn.Linear(ff_dim * seq_len, 512),  # Linear layer
            #nn.ReLU(),  # ReLU activation
            # nn.Linear(embedding_dim, 256),  # Linear layer
            # nn.ReLU(),  # ReLU activation
            nn.Linear(embedding_dim, 128),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(128, 3)  # Output layer for regression
        )

    def forward(self, inputs, return_attn=False):
        x = self.encoder_input_layer(inputs)

        x = self.positional_encoding(x)

        attention_matrices = []
        for attention_layer in self.attention_layers:
            x, attention_scores = attention_layer(x)
            if return_attn:
                attention_matrices.append(attention_scores)

        x = self.norm(x)

        x=x.mean(dim=1)

        x = self.final_layers(x)

        return x


"""
class TransformerDecoder(nn.Module):
    def __init__(self, num_voxels, classes, time2vec_dim, num_heads=1, head_size=128, ff_dim=None, num_layers=1,
                 dropout=0):
        super(TransformerDecoder, self).__init__()

        # Set ff_dim if not provided
        if ff_dim is None:
            ff_dim = head_size

        # Define the positional encoding
        self.positional_encoding = PositionalEncoding(dropout=dropout, max_seq_len=seq_len, d_model=embedding_dim,
                                                      batch_first=False)

        # Create the attention layers
        self.attention_layers = nn.ModuleList(
            [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in
             range(num_layers)]
        )

        # Layer normalization after all attention layers
        self.norm = nn.LayerNorm(ff_dim)  # Normalize the output after the last attention block

        # Final feed-forward layers
        self.final_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the input
            nn.Linear(ff_dim * seq_len, 512),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(512, 256),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(256, 128),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(128, classes)  # Output layer for classification
        )

    def forward(self, memory, tgt, return_attn=False):
        # `memory` is the encoder's output, `tgt` is the target sequence for decoding
        tgt = self.positional_encoding(tgt)  # Apply positional encoding to target

        attention_matrices = []  # List to store attention matrices if required

        # Pass through each attention layer
        for attention_layer in self.attention_layers:
            tgt, attention_scores = attention_layer(tgt)  # Apply attention block
            if return_attn:
                attention_matrices.append(attention_scores)

        # Apply layer normalization after the last attention block
        tgt = self.norm(tgt)

        # Pass through final layers to get the output
        output = self.final_layers(tgt)

        return output

# Full Transformer Model (Encoder + Decoder)
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, num_voxels, classes, time2vec_dim, num_heads=1, head_size=128, ff_dim=None, num_layers=1,
                 dropout=0):
        super(TransformerModel, self).__init__()

        # Encoder initialization
        self.encoder = TransformerEncoder(
            num_voxels=num_voxels,  # Input features (num_voxels)
            classes=classes,  # Output classes (for classification)
            time2vec_dim=time2vec_dim,  # Time embedding dimension (if needed)
            num_heads=num_heads,  # Attention heads
            head_size=head_size,  # Head size for multi-head attention
            ff_dim=ff_dim,  # Feed-forward dimension (if provided)
            num_layers=num_layers,  # Number of encoder layers
            dropout=dropout  # Dropout for regularization
        )

        # Decoder initialization
        self.decoder = TransformerDecoder(
            num_voxels=ff_dim,  # Input features (num_voxels)
            classes=classes,  # Output classes (for classification)
            time2vec_dim=time2vec_dim,  # Time embedding dimension (if needed)
            num_heads=num_heads,  # Attention heads
            head_size=head_size,  # Head size for multi-head attention
            ff_dim=ff_dim,  # Feed-forward dimension (if provided)
            num_layers=num_layers,  # Number of encoder layers
            dropout=dropout  # Dropout for regularization
        )

    def forward(self, x, tgt=None):
        # Forward pass through the encoder
        memory = self.encoder(x)  # Encoder processes the input sequence (x)

        # If target is provided, pass it to the decoder
        if tgt is not None:
            output = self.decoder(memory, tgt)  # Decoder processes the memory from encoder and target (tgt)
        else:
            output = memory  # In case of no target, use encoder's output directly

        return output

"""