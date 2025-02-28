import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass, field

class EEG_Embed(nn.Module):

    def __init__(self, kernel_size=3, F1=8, D=8, F2=None, n_emb=128, hidden=128, AP1 = 4, AP2 = 8, dropout = 0.2, dropout_cslf = 0.4, input_channels=64, input_len=320):
        
        super().__init__()

        if F2 is None: F2 = F1 * D

        self.input_channels = input_channels
        self.input_normalization = nn.BatchNorm1d(input_channels)

        # input with shape (B, 1, T, C) => outputs (B, F1, T, C)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0)), # temporal convolution => model intra-channel dependencies
            nn.BatchNorm2d(F1) # 1st batch norm
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1*D, (1, input_channels), groups=F1), # spatial convolution => channel dimension to 1 model inter-channel dependencies
            nn.BatchNorm2d(F1*D), # 2nd batch norm
            nn.ReLU(), # act. function
            nn.AvgPool2d((AP1, 1)), # average pooling by a factor of 2
            nn.Dropout2d(p=dropout) # dropout
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, kernel_size=3, padding=1, groups=F1*D), # depthwise separable convolution
            nn.Conv2d(F1*D, F2, kernel_size=1), # pointwise
            nn.BatchNorm2d(F2), # 3rd batch norm
            nn.ReLU(), 
            nn.AvgPool2d((AP2, 1)), # adaptative pooling to reduce T dim to 1
            nn.Dropout2d(p=dropout)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * (input_len // (AP1 * AP2)), hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_cslf),
            nn.Linear(hidden, n_emb)
        )

    def forward(self, x):
        
        # input shape must be (batch, n_chan, n_samples)
        assert x.shape[1] == self.input_channels, f'In put must be of shape (B, C, T) with C={self.input_channels} and {x.shape[1]} were introduced'

        x = self.input_normalization(x)

        # add feature dimension and transpose 
        x = x.transpose(1, 2) # (B, C, T) => (B, T, C)
        x = x.unsqueeze(1) # (B, T, C) => (B, 1, T, C)
        x = self.temporal(x) # (B, 1, T, C) => (B, F1, T, C)
        x = self.spatial(x) # (B, F1, T, C) => (B, F1*D, T', 1)
        x = self.depthwise(x) # (B, F1*D, T', 1) => (B, F2, T'', 1)
        x = torch.flatten(x, start_dim=1) # (B, F2, T'', 1) => (B, F2*T'')
        x = self.fc(x) # (B, F2*T'') => (B, n_emb)
        return x

class Stim_Embed_LSTM(nn.Module):

    def __init__(self, hidden_dim=64, n_layers=2, n_emb=128):
        
        super().__init__()

        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_emb)

        # Normalization layer
        self.bn_input = nn.BatchNorm1d(1)
        
    def forward(self, x):
        
        if len(x.shape) == 2: x = x.unsqueeze(1)

        x = self.bn_input(x)

        if x.shape[1] == 1: x = x.transpose(1, 2) # x with shape (B, 1, T) => (B, T, 1)

        _, (h_n, _) = self.lstm(x) # tomar el último estado oculto (h_n)
        embedding = self.fc(h_n[-1]) # proyectar a la dimensión del embedding
        return embedding
    
class Stim_Embed_Conv(nn.Module):

    def __init__(self, kernel_size=5, AP1=4, AP2=2, n_emb=128, hidden=128, input_len=320, dropout=0.2, dropout_clsf=0.4):

        super().__init__()

        self.convnet = nn.ModuleList([
            nn.Conv1d(1, 8, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
            nn.AvgPool1d(kernel_size=AP1),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 16, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
            nn.AvgPool1d(kernel_size=AP2),
            nn.BatchNorm1d(16)
        ])

        # Normalization layer
        self.bn_input = nn.BatchNorm1d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * (input_len // (AP1 * AP2)), hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_clsf),
            nn.Linear(hidden, n_emb)
        )

    def forward(self, x):

        if len(x.shape) == 2: x = x.unsqueeze(1) # (B, T) => (B, 1, T)

        x = self.bn_input(x)

        for layer in self.convnet:# (B, 1, T)
            x = layer(x)

        x = self.fc(x) # (B, 16, ) => (B, 16, )

        return x
    
# class Stim_Embed_TCN(nn.Module):

#     def __init__(self, kernel_size=3, n_blocks, n_emb=128):

#         super().__init__()

#         self.embedding = nn.Conv1d(1, emb_size)

#         self.block_1 = nn.ModuleList([
#             nn.Conv1d(1, 8, 1),
#             nn.Bat
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#             nn.Conv1d(8, 16, kernel_size, stride=kernel_size//2, padding=kernel_size//2),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#         ])

#         # Normalization layer
#         self.bn_input = nn.BatchNorm1d(1)

#         self.fc = nn.Linear(64, n_emb)

#     def forward(self, x):

#         if len(x.shape) == 2: x = x.unsqueeze(1)

#         x = self.bn_input(x)

#         for layer in self.convnet:
#             x = layer(x)

#         x = torch.flatten(x, start_dim=1) # (B, 128, 1) => (B, 128)
#         x = self.fc(x) # (B, 128) => (B, 128)

#         return x
    
@dataclass
class Triplet_Net_Config:

    # Global
    n_emb:int = 128
    stim_emb:str = 'Conv'

    # EEG embedder
    eeg_ks:int = 3
    eeg_F1:int = 32
    eeg_D:int = 8
    eeg_AP1:int = 4
    eeg_chan:int = 64
    eeg_samples:int = 320

    # Stim LSTM embedder
    stim_lstm_hidden:int = 64
    stim_lstm_layers:int = 2

    # Stim Conv1d embedder
    stim_conv_ks:int = 5

    # Classifier
    out_dim:int = 1

    # Dropout
    dropout = 0.2
    dropout_clsf = 0.4

    
class Triplet_Net(nn.Module):

    def __init__(self, config:Triplet_Net_Config):

        super().__init__()

        self.eeg_embeder = EEG_Embed(kernel_size=config.eeg_ks, 
            n_emb=config.n_emb, 
            F1=config.eeg_F1, 
            D=config.eeg_D, 
            AP1=config.eeg_AP1, 
            dropout=config.dropout,
            dropout_cslf=config.dropout_clsf,
            input_channels=config.eeg_chan, 
            input_len= config.eeg_samples
        )

        if config.stim_emb == 'LSTM':
            self.stim_embeder = Stim_Embed_LSTM(hidden_dim=config.stim_lstm_hidden, 
                n_layers=config.stim_lstm_layers, 
                n_emb=config.n_emb
            )
        # elif config.stim_emb == 'CNN':
        #     self.stim_embeder = Stim_Embed_LSTM(hidden_dim=config.stim_lstm_hidden, 
        #         n_layers=config.stim_lstm_layers, 
        #         n_emb=config.n_emb
        #     )
        else:
            self.stim_embeder = Stim_Embed_Conv(kernel_size=config.stim_conv_ks,
                n_emb=config.n_emb,
                dropout=config.dropout,
                dropout_clsf=config.dropout_clsf
            )

        self.classifier = nn.Linear(config.n_emb, config.out_dim)

    def forward(self, eeg, stima, stimb):     

        emb_eeg = self.eeg_embeder(eeg)
        emb_stima = self.stim_embeder(stima)
        emb_stimb = self.stim_embeder(stimb)

        return emb_eeg, emb_stima, emb_stimb


