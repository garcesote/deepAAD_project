import torch
import torch.nn as nn
import numpy as np

# model description
class FCNN(nn.Module):

    def __init__(self, n_hidden, dropout=0.45, n_chan=63, n_samples=50, output_dim=1):
        
        super().__init__()
        self.input_dim = n_chan * n_samples
        self.n_chan = n_chan
        self.n_samples = n_samples
        self.output_dim = output_dim
        layers = []

        # First flatten layer (B, T x C)
        layers.append(nn.Flatten(1, -1))

        # When predicting window calculate the prearsonr mean
        self.window_pred = True if self.n_samples == self.output_dim else False

        # Hidden layers
        for k in range(n_hidden):
            layers.append(nn.Linear(int(self.input_dim - self.input_dim / (n_hidden+1) * k), 
                            int(self.input_dim - self.input_dim / (n_hidden+1) * (k+1)))) # layers with desired neurons
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p = dropout))

        # Output layer
        layers.append(nn.Linear(int(self.input_dim / (n_hidden+1)), self.output_dim))
        
        self.model =  nn.Sequential(*layers) # * para introducir los elementos de la lista por separado (si no esta un elemento con la lista)

    def forward(self, input):
        # input shape must be (batch, n_chan, n_samples)
        if list(input.shape) == [input.shape[0], self.n_chan, self.n_samples]:
            preds = self.model(input)
            return preds
        else:
            raise ValueError("Se debe introducir un tensor con las dimensiones adecuadas (B, C, T)")
    
    # Freeze all layers except the last one
    def finetune(self):
        for layer in self.model[:-1]:
            for param in layer.parameters():
                param.requires_grad = False


class CNN(nn.Module):

    def __init__(self, F1=8, D=8, F2=None, AP1 = 2, AP2 = 4, dropout = 0.2, input_channels=64, input_samples=50, output_dim=1, post_stim = None):

        super().__init__()

        if F2 is None: F2 = F1 * D
        self.F1 = F1
        self.F2 = F2
        self.post_stim = post_stim

        # When predicting window calculate the prearsonr mean
        self.window_pred = True if input_samples == output_dim else False

        if post_stim:
            assert output_dim == 1, 'When using post stim mode follow sample pred methodology (output_dim=1)'
            if post_stim == 'narrow': # eeg signal ranging from 100ms to 250ms post-stim (10 samples)
                input_samples = 10 
            elif post_stim == 'broad': # eeg signal ranging from 100ms to 400ms post-stim (20 samples)
                input_samples = 20
            else:
                raise ValueError('post_stim must be a broad/narrow value')
            
        self.input_channels = input_channels
        self.input_samples = input_samples
        self.output_dim = output_dim

        # input with shape (B, 1, T, C) => outputs (B, F1, T, C)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(3, 1), padding=(1,0)), # temporal convolution => model intra-channel dependencies
            nn.BatchNorm2d(F1) # 1st batch norm
        )
        
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1*D, (1,input_channels), groups=F1), # spatial convolution => channel dimension to 1 model inter-channel dependencies
            nn.BatchNorm2d(F1*D), # 2nd batch norm
            nn.ELU(alpha=1.0), # act. function
            nn.AvgPool2d((AP1, 1)), # average pooling by a factor of 2
            nn.Dropout2d(p=dropout) # dropout
        )
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, kernel_size=3, padding=1, groups=F1*D), # depthwise separable convolution
            nn.Conv2d(F1*D, F2, kernel_size=1), # pointwise
            nn.BatchNorm2d(F2), # 3rd batch norm
            nn.ELU(alpha=1.0), 
            nn.AvgPool2d((AP2, 1)), # average pooling by a factor of 5
            nn.Dropout2d(p=dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim = 1, end_dim = -1), # concat feature and sample dimension
            nn.Linear(F2 * (input_samples // (AP1 * AP2)), self.output_dim, bias=True) # apply linear layer to obtain the unit output
        )
    
    def forward(self, input):

        if self.post_stim == 'narrow':
            input = input[:, :, 6:16]
        elif self.post_stim == 'broad':
            input = input[:, :, 6:26]

        # input shape must be (batch, n_chan, n_samples)
        assert list(input.shape) == [input.shape[0], self.input_channels, self.input_samples], 'In put must be of shape (B, C, T)'

        # add feature dimension and transpose 
        input = input.transpose(1, 2) # (B, C, T) => (B, T, C)
        x = input.unsqueeze(1) # (B, T, C) => (B, 1, T, C)
        x = self.temporal(x) # (B, 1, T, C) => (B, F1, T, C)
        x = self.spatial(x) # (B, F1, T, C) => (B, F1*D, T, 1)
        x = self.depthwise(x) # (B, F1*D, T, 1) => (B, F2, T, 1)
        preds = self.classifier(x) # (B, F2, T, 1) => (B, F2*T) => (B, n_out)
        return preds
        
    # Freeze the temporal, spatial and depthwise blocks
    def finetune(self):
        for param in self.temporal.parameters():
            param.requires_grad = False
        for param in self.spatial.parameters():
            param.requires_grad = False
        for param in self.depthwise.parameters():
            param.requires_grad = False