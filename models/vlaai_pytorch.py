import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    def __init__(self, filters=(256, 256, 256, 128, 128), kernels=(8,)*5, input_channels=64):
        super(Extractor, self).__init__()
        self.layers = nn.ModuleList()
        for filter_, kernel in zip(filters, kernels):
            self.layers.append(nn.Conv1d(input_channels, filter_, kernel))
            self.layers.append(nn.LeakyReLU())  # LayerNorm added later in the forward method
            self.layers.append(nn.ConstantPad1d((0, kernel - 1), 0))
            input_channels = filter_
            
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply LayerNorm after each LeakyReLU
            if isinstance(layer, nn.LeakyReLU):
                norm_shape = x.shape[1:]
                x = nn.LayerNorm(norm_shape).to(x.device)(x)
        return x


class OutputContext(nn.Module):
    def __init__(self, filter_=64, kernel=32, input_channels=64):
        super(OutputContext, self).__init__()
        self.pad = nn.ConstantPad1d((kernel - 1, 0), 0)
        self.conv = nn.Conv1d(input_channels, filter_, kernel)
        self.activation = nn.LeakyReLU()
        # LayerNorm added later in the forward method

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        norm_shape = x.shape[1:]
        x = nn.LayerNorm(norm_shape).to(x.device)(x)
        return self.activation(x)

class VLAAI(nn.Module):
    def __init__(self, n_blocks=4, input_channels=64, output_dim=1, use_skip=True, extractor_output = 128):
        super(VLAAI, self).__init__()
        self.n_blocks = n_blocks
        self.use_skip = use_skip
        self.extractor = Extractor(input_channels=input_channels)
        self.dense = nn.Linear(extractor_output, input_channels)  # Equivalent of Dense in TF
        self.output_context = OutputContext(input_channels=input_channels)
        self.final_dense = nn.Linear(input_channels, output_dim)

    def forward(self, x):
        for _ in range(self.n_blocks):
            skip = x if self.use_skip else 0
            x = self.extractor(x + skip)
            x = x.transpose(1, 2)
            x = self.dense(x)
            x = x.transpose(1, 2)
            x = self.output_context(x)
        x = x.transpose(1, 2)
        x = self.final_dense(x)
        x = x.transpose(1, 2)
        x = torch.squeeze(x, dim=1)
        return x