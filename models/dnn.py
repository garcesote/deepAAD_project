import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import scipy
from utils.functional import get_loss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

    def __init__(self, F1=8, D=8, F2=64, AP1 = 2, AP2 = 4, dropout = 0.2, input_channels=64, input_samples=50, output_dim=1, aad_classifier=None):

        super().__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.input_channels = input_channels
        self.input_samples = input_samples
        self.output_dim = output_dim

        # When predicting window calculate the prearsonr mean
        self.window_pred = True if self.input_samples == self.output_dim else False

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

        if aad_classifier is not None: self.aad_classifier == LinearDiscriminantAnalysis()
    
    def forward(self, input):
        # input shape must be (batch, n_chan, n_samples)
        if list(input.shape) == [input.shape[0], self.input_channels, self.input_samples]:

            # add feature dimension and transpose 
            input = input.transpose(1, 2) # (B, C, T) => (B, T, C)
            x = input.unsqueeze(1) # (B, T, C) => (B, 1, T, C)
            x = self.temporal(x) # (B, 1, T, C) => (B, F1, T, C)
            x = self.spatial(x) # (B, F1, T, C) => (B, F1*D, T, 1)
            x = self.depthwise(x) # (B, F1*D, T, 1) => (B, F2, T, 1)
            preds = self.classifier(x) # (B, F2, T, 1) => (B, F2*T) => (B, n_out)
            return preds
        
        else:
            raise ValueError('El input de la red tiene que guardar dimensiones (B, C, T)')
    
    def fit_classifier(self, eeg, stima, stimb, batch_size):

        """ Fit the classifier model using the CCA coefficients
        
        Args:

            eeg (array-like or tensor): EEG signal of shape (n_channels, n_samples)

            stima (array-like or tensor): attended envelope of shape (n_samples, )

            stimb (array-like or tensor): unattended envelope of shape (n_samples, )

            batch_size (int): lenght of the windowed data

        """

        # Get the scores
        preds = self(eeg)

        scores_a = self._compute_correlation(preds, stima)
        scores_b = self._compute_correlation(preds, stimb)

        # Difference between scores as the function to classify
        # f_att = scores_a - scores_b
        # f_unatt = scores_b - scores_a
        f_att = scores_a
        f_unatt = scores_b

        # Concatenate the scores and generate a label array to feed LDA
        scores = np.vstack((f_att, f_unatt))
        labels = np.concatenate((np.ones(f_att.shape[0]), np.zeros(f_unatt.shape[0])))

        self.classf.fit(scores, labels)

    # Compute the pearson correlation coefficient
    def _compute_correlation(self, preds, targets, eps):

        # Compute the correlation for all channels or batches
        n_stim, n_samples = preds.shape
        corr = torch.zeros((n_stim, ))
        for chan, (p, t) in enumerate(zip(preds, targets)):
            vx = p - torch.mean(p)
            vy = t - torch.mean(t)
            corr[chan] = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
        return corr
    
    # Compute the interaural level difference ILD (dB)
    def _compute_ild(self, left_channel, right_channel):
        # Calculate RMS for each channel
        rms_left = torch.sqrt(torch.mean(left_channel**2))
        rms_right = torch.sqrt(torch.mean(right_channel**2))
        # Calculate ILD in dB
        ild = 10 * torch.log10(rms_left / rms_right)
        return ild
        
    # Freeze the temporal, spatial and depthwise blocks
    def finetune(self):
        for param in self.temporal.parameters():
            param.requires_grad = False
        for param in self.spatial.parameters():
            param.requires_grad = False
        for param in self.depthwise.parameters():
            param.requires_grad = False