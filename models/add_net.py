import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field

class Inception_Block(nn.Module):

    def __init__(self, input_channels:int = 64, out_transform:int = 32, out_feat:int = 8, feature_proj:list = [16,8,4,2], feature_kernel:list = [19,25,33,39], pool:int = 3):

        super().__init__()

        assert len(feature_kernel) == len(feature_proj), "The list containing the kernels and channels of the features must be of the same size"
        self.n_feat = len(feature_proj)
        self.out_feat = out_feat
        self.input_channels = input_channels
        self.pool = pool

        # Output channels obtained after concatenate all the outputs of each branch
        if pool:
            self.concat_channels = (self.n_feat + 1) * out_feat + out_transform
        else:
            self.concat_channels = self.n_feat * out_feat + out_transform

        # Transform branch from input channels to out_transform channels to learn spatial features
        self.transform = nn.Conv1d(input_channels, out_transform, kernel_size=1)

        # For the feature extractor one list corresponding to the conv modules that reduce the chan dim
        self.chan_compressor = nn.ModuleList([
            nn.Conv1d(input_channels, chan, kernel_size=1)
        for chan in feature_proj])
        # And an other list containing the conv modules that implement the kernel to learn temporal features
        self.feat_extractor = nn.ModuleList([
            nn.Conv1d(chan, out_feat, kernel_size=kernel, padding='same')
        for kernel, chan in zip(feature_kernel, feature_proj)])

        # Branch containing the pooling operation
        if pool:
            self.pool = nn.MaxPool1d(kernel_size=pool, stride=1, padding=1)
            self.pool_compressor = nn.Conv1d(input_channels, out_feat, kernel_size=1)

        # Batch norm layers
        self.input_norm = nn.BatchNorm1d(num_features=input_channels)
        self.out_norm = nn.BatchNorm1d(num_features=self.concat_channels)

    def forward(self, x):
        
        assert x.shape[1] == self.input_channels, f"EEG input with size (B, C, T) must have {self.input_channels} and {input.shape[1]} were introduced"
        B, C, T = x.size()

        # Batch normalization
        x_norm = self.input_norm(x)

        # Transform branch: (B, C, T) => (B, out_transform, T)
        x_transform = self.transform(x_norm)

        # Feature branches
        x_feat = []
        for n in range(self.n_feat):
            # Channel compression
            x_n = self.chan_compressor[n](x_norm) # (B, C, T) => (B, feature_proj[n], T)
            x_n = self.feat_extractor[n](x_n) # (B, feature_proj[n], T) => (B, out_feat, T)
            x_feat.append(x_n)

        # Concatenate the result of the branches
        x_concat = [x_transform] + x_feat

        if self.pool:
            # Pool branch
            x_pool = self.pool(x_norm)
            x_pool = self.pool_compressor(x_pool)
            x_concat = x_concat + [x_pool]

        x_concat = torch.cat(x_concat, dim = 1) # (B, concat_channels, T)

        # Final batch normalization
        x_concat = self.out_norm(x_concat)

        return x_concat
    
class MLP_Classifier(nn.Module):

    def __init__(self, input_size, out_dim, hidden_size, dropout):
        
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(self, x):

        return self.classifier(x)
    
@dataclass
class AAD_Net_Config:

    # EEG Inception Block
    eeg_chan:int = 64
    eeg_transform:int = 32
    eeg_out_feat:int = 8
    eeg_feat_proj:list = field(default_factory=lambda: [16, 8, 4, 2])
    eeg_feat_kernel:list = field(default_factory=lambda: [19, 25, 33, 39])
    eeg_pool:int = 3

    # Stim Inception Block
    stim_chan:int = 1
    stim_transform:int = 1
    stim_out_feat:int = 4
    stim_feat_proj:list = field(default_factory=lambda: [1, 1])
    stim_feat_kernel:list = field(default_factory=lambda: [65, 81])
    stim_pool:int = None

    # Classifier
    out_dim:int = 2
    hidden_size:int = 16
    dropout:float = 0.4
    
class AAD_Net(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.eeg_inception = Inception_Block(config.eeg_chan, config.eeg_transform, config.eeg_out_feat, config.eeg_feat_proj, config.eeg_feat_kernel, config.eeg_pool)
        
        self.stim_inception = Inception_Block(config.stim_chan, config.stim_transform, config.stim_out_feat, config.stim_feat_proj, config.stim_feat_kernel, config.stim_pool)

        self.input_clsf_size = self.eeg_inception.concat_channels * self.stim_inception.concat_channels * 2
        self.classifier = MLP_Classifier(self.input_clsf_size, config.out_dim, config.hidden_size, config.dropout)

    def forward(self, eeg, stima, stimb):

        # EEG and stima with shape (B, C, T)
        assert eeg.shape[-1] == stima.shape[-1] and eeg.shape[-1] == stimb.shape[-1], f"Input eeg with {eeg.shape[-1]} samples must be of the same lenght as the stim"

        eeg_series = self.eeg_inception(eeg) # (B, Ce, T)

        stima_series = self.stim_inception(stima) # (B, Cs, T)
        stimb_series = self.stim_inception(stimb) # (B, Cs, T)

        # Stack the stim series
        stim_series = torch.hstack((stima_series, stimb_series)) # (B, Cs*2, T)

        # Broadcast and concat the stim time series
        eeg_broad = eeg_series.unsqueeze(1) # (B, 1, Ce, T)
        stim_broad = stim_series.unsqueeze(2) # (B, Cs*2, 1, T)

        # Compute the Pearsonr coeficients
        coefs = self.compute_correlation(eeg_broad, stim_broad) # (B, 2*Cs, Ce)
        coefs = torch.flatten(coefs, start_dim=1) # (B, 2*Cs*Ce)

        # Classify the coefficients and apply Softmax for probs
        logits = self.classifier(coefs)
        preds = torch.softmax(logits, dim=1)

        return preds

    def compute_correlation(self, preds, targets, eps=1e-8):

        """
        Compute Pearson correlation coefficient for all channels in a batch.
        Args:
            preds (torch.Tensor): Predicted values (:, samples).
            targets (torch.Tensor): Target values (:, samples).
            eps (float): Small value to prevent division by zero.
        Returns:
            torch.Tensor: Correlation coefficients for each channel.
        """

        # Compute means
        preds_mean = preds.mean(dim=-1, keepdim=True)
        targets_mean = targets.mean(dim=-1, keepdim=True)
        
        # Compute deviations
        preds_dev = preds - preds_mean
        targets_dev = targets - targets_mean
        
        # Compute correlation for all channels simultaneously
        numerator = (preds_dev * targets_dev).sum(dim=-1)
        denominator = torch.sqrt((preds_dev**2).sum(dim=-1) * (targets_dev**2).sum(dim=-1)) + eps
        return numerator / denominator
    
    # Freeze inception blocks when finetunning
    def finetune(self):
        for param in self.stim_inception.parameters():
            param.requires_grad=False
        for param in self.eeg_inception.parameters():
            param.requires_grad = False
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)