import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import scipy
from utils.functional import correlation

# model description

class FCNN(nn.Module):

    def __init__(self, n_hidden, dropout=0.45, n_chan=63, n_samples=50):
        
        super().__init__()
        self.input_dim = n_chan * n_samples
        self.n_chan = n_chan
        self.n_samples = n_samples
        layers = []

        # First flatten layer (B, T x C)
        layers.append(nn.Flatten(1, -1))

        # Hidden layers
        for k in range(n_hidden):
                layers.append(nn.Linear(int(self.input_dim - self.input_dim / (n_hidden+1) * k), 
                              int(self.input_dim - self.input_dim / (n_hidden+1) * (k+1)))) # layers with desired neurons
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(p = dropout))

        # Output layer
        layers.append(nn.Linear(int(self.input_dim / (n_hidden+1)), 1))
        
        self.model =  nn.Sequential(*layers) # * para introducir los elementos de la lista por separado (si no esta un elemento con la lista)

    def forward(self, input, targets=None):
        # input shape must be (batch, n_chan, n_samples)
        if list(input.shape) == [input.shape[0], self.n_chan, self.n_samples]:
            preds = torch.squeeze(self.model(input)) # generates an output with shape (batch, 1) => for each batch generates a single envelope prediction
            if targets is None:
                loss = None
            else:
                loss = - correlation(preds, targets)
            return preds, loss
        else:
            raise ValueError("Se debe introducir un tensor con las dimensiones adecuadas (B, C, T)")
        

class CNN(nn.Module):

    def __init__(self, F1=8, D=8, F2=64, AP1 = 2, AP2 = 4, dropout = 0.2, input_channels=64, input_samples=50):

        super().__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.input_channels = input_channels
        self.input_samples = input_samples

        layers = []
        
        # input with shape (B, 1, T, C) => outputs (B, F1, T, C)
        layers.append(nn.Conv2d(1, F1, kernel_size=(3, 1), padding=(1,0))) # temporal convolution => model intra-channel dependencies
        layers.append(nn.BatchNorm2d(F1)) # 1st batch norm
        
        layers.append(nn.Conv2d(F1, F1*D, (1,input_channels), groups=F1)) # spatial convolution => channel dimension to 1 model inter-channel dependencies
        layers.append(nn.BatchNorm2d(F1*D)) # 2nd batch norm
        layers.append(nn.ELU(alpha=1.0)) # act. function
        layers.append(nn.AvgPool2d((AP1, 1))) # average pooling by a factor of 2
        layers.append(nn.Dropout2d(p=dropout)) # dropout
        
        layers.append(nn.Conv2d(F1*D, F1*D, kernel_size=3, padding=1, groups=F1*D)) # depthwise separable convolution
        layers.append(nn.Conv2d(F1*D, F2, kernel_size=1)) # pointwise
        layers.append(nn.BatchNorm2d(F2)) # 3rd batch norm
        layers.append(nn.ELU(alpha=1.0)) 
        layers.append(nn.AvgPool2d((AP2, 1))) # average pooling by a factor of 5
        layers.append(nn.Dropout2d(p=dropout))

        layers.append(nn.Flatten(start_dim = 1, end_dim = -1)) # concat feature and sample dimension
        layers.append(nn.Linear(int(F2 * (input_samples / (AP1 * AP2))), 1, bias=True)) # apply linear layer to obtain the unit output

        self.model = nn.Sequential(*layers)
    
    def forward(self, input, targets = None):
        # input shape must be (batch, n_chan, n_samples)
        if list(input.shape) == [input.shape[0], self.input_channels, self.input_samples]:

            # add feature dimension and 
            input = input.transpose(1, 2)
            input = input.unsqueeze(1) 

            preds = torch.squeeze(self.model(input))
            if targets is None:
                loss = None
            else:
                loss = -correlation(targets, preds)
            return preds, loss
        else:
            raise ValueError('El input de la red tiene que guardar dimensiones (B, C, T)')