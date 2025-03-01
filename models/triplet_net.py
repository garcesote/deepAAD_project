import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass, field
import numpy as np

class EEG_Embed(nn.Module):
    def __init__(self, kernel_size=3, F1=16, D=4, F2=None, n_emb=128, hidden=128,  AP1=4, AP2=8,
                dropout=0.3, dropout_cslf=0.5, input_channels=64, input_len=320, l2_reg=1e-4):
        
        super().__init__()

        if F2 is None: F2 = F1 * D

        self.input_channels = input_channels
        self.input_normalization = nn.BatchNorm1d(input_channels)
        self.l2_reg = l2_reg

        # Temporal convolution
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0)),
            nn.BatchNorm2d(F1),
            nn.ELU()  # Using ELU instead of ReLU for better gradients
        )

        # Spatial convolution
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1*D, (1, input_channels), groups=F1),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((AP1, 1)),
            nn.Dropout2d(p=dropout)
        )

        # Depthwise separable convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, kernel_size=3, padding=1, groups=F1*D),
            nn.Conv2d(F1*D, F2, kernel_size=1),
            nn.BatchNorm2d(F2),
            nn.ELU(), 
            nn.AvgPool2d((AP2, 1)),
            nn.Dropout2d(p=dropout)
        )
        
        # Fully connected layers with residual connection
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(F2 * (input_len // (AP1 * AP2)), hidden)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(p=dropout_cslf)
        self.fc2 = nn.Linear(hidden, n_emb)
        
        # Optional: Batch normalization for embeddings
        self.bn_emb = nn.BatchNorm1d(n_emb)

    def forward(self, x):
        assert x.shape[1] == self.input_channels, f'Input must be of shape (B, C, T) with C={self.input_channels}'

        x = self.input_normalization(x)
        x = x.transpose(1, 2)  # (B, C, T) => (B, T, C)
        x = x.unsqueeze(1)     # (B, T, C) => (B, 1, T, C)
        
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.depthwise(x)
        
        x = self.flatten(x)
        x1 = self.fc1(x)
        x1 = self.act(x1)
        x1 = self.dropout(x1)
        x = self.fc2(x1)
        
        # L2 normalization of embeddings (optional but recommended for triplet loss)
        if self.l2_reg > 0:
            x = F.normalize(x, p=2, dim=1)
        
        return x

class Stim_Embed_Conv(nn.Module):
    def __init__(self, kernel_size=5, features = [16, 32], pool = [2, 4], n_emb=128, hidden=128, 
                 input_len=320, dropout=0.3, dropout_clsf=0.5, l2_reg=1e-4, skip=True):

        super().__init__()
        
        self.l2_reg = l2_reg
        
        # Input normalization
        self.bn_input = nn.BatchNorm1d(1)

        in_features = [1] + features[:-1]
        print
        
        # Convolutional layers
        if not skip:
            self.cnn_stack = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(in_f, out_f, kernel_size, padding='same'),
                    nn.BatchNorm1d(out_f),
                    nn.ELU(),
                    nn.AvgPool1d(kernel_size=p),
                    nn.Dropout1d(p=dropout),
                )
                for p, in_f, out_f in zip(pool, in_features, features)
            ])

            self.flatten_dim = features[-1] * (input_len // np.prod(pool))
            
        else:
            # Project the stim into a space with n features
            self.proj = nn.Sequential(
                nn.Conv1d(1, features[-1], kernel_size, padding='same'),
            )
            # Conv blocks
            self.cnn_stack = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(features[-1], features[-1], kernel_size, padding='same'),
                    nn.BatchNorm1d(features[-1]),
                    nn.ELU(),
                    nn.Dropout1d(p=dropout),
                )
                for _ in range(len(features))
            ])

            self.flatten_dim = features[-1] * (input_len)
        
        # Fully connected layers
        self.fc = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(self.flatten_dim, hidden),
            nn.ELU(),
            nn.Dropout(p=dropout_clsf),
            nn.Linear(hidden, n_emb),
        ])
        
        self.skip = skip
        
        # Optional: Batch normalization for embeddings
        self.bn_emb = nn.BatchNorm1d(n_emb)

    def forward(self, x):
        if len(x.shape) == 2: 
            x = x.unsqueeze(1)  # (B, T) => (B, 1, T)

        x = self.bn_input(x)
        
        # Conv blocks
        if self.skip:
            x = self.proj(x)
            for layer in self.cnn_stack:
                x = x + layer(x)
        else:
            for layer in self.cnn_stack:
                x = layer(x)
        
        # Fully connected layers
        for layer in self.fc:
            x = layer(x)
        
        # L2 normalization of embeddings (optional but recommended for triplet loss)
        if self.l2_reg > 0:
            x = F.normalize(x, p=2, dim=1)
        
        return x
    
@dataclass
class Triplet_Net_Config:
    # Global
    n_emb: int = 128
    margin: float = 0.3  # Margin for triplet loss
    
    # EEG embedder
    eeg_ks: int = 3
    eeg_F1: int = 8    # Reduced from 32
    eeg_D: int = 8      # Reduced from 8
    eeg_AP1: int = 4
    eeg_chan: int = 64
    eeg_samples: int = 320
    
    # Stim Conv1d embedder
    stim_conv_ks: int = 5
    stim_features:list = field(default_factory=lambda: [16, 32])
    stim_pool:list = field(default_factory=lambda: [4, 2])
    skip:bool = True
    
    # Dropout
    dropout: float = 0.3    # Increased from 0.2
    dropout_clsf: float = 0.5  # Increased from 0.4
    
    # Regularization
    l2_reg: float = 1e-4
    
class Triplet_Net(nn.Module):
    def __init__(self, config: Triplet_Net_Config):
        super().__init__()
        
        self.margin = config.margin

        self.eeg_embeder = EEG_Embed(
            kernel_size=config.eeg_ks, 
            n_emb=config.n_emb, 
            hidden=config.n_emb,
            F1=config.eeg_F1, 
            D=config.eeg_D, 
            AP1=config.eeg_AP1, 
            dropout=config.dropout,
            dropout_cslf=config.dropout_clsf,
            input_channels=config.eeg_chan, 
            input_len=config.eeg_samples,
            l2_reg=config.l2_reg
        )

        self.stim_embeder = Stim_Embed_Conv(
            kernel_size=config.stim_conv_ks,
            features=config.stim_features,
            hidden=config.n_emb,
            pool = config.stim_pool,
            n_emb=config.n_emb,
            dropout=config.dropout,
            dropout_clsf=config.dropout_clsf,
            l2_reg=config.l2_reg,
            skip=config.skip
        )
        
    def forward(self, eeg, stima, stimb):     
        emb_eeg = self.eeg_embeder(eeg)
        emb_stima = self.stim_embeder(stima)
        emb_stimb = self.stim_embeder(stimb)

        return emb_eeg, emb_stima, emb_stimb
    
    def compute_triplet_loss(self, eeg, stima, stimb):
        
        # Get embeddings
        emb_eeg, emb_stima, emb_stimb = self.forward(eeg, stima, stimb)
        
        # Calculate distances
        pos_dist = F.pairwise_distance(emb_eeg, emb_stima)
        neg_dist = F.pairwise_distance(emb_eeg, emb_stimb)
        
        # Triplet loss with margin
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        
        # Return mean loss and also individual distances for monitoring
        return losses.mean(), pos_dist.mean(), neg_dist.mean()