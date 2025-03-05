import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm_perm(nn.Module):

    """ LayerNorm_perm: module that perform a layer normalization tranformation permuting the data data to perform it over the channel dim
    
    Parameters
    ---------
    features: int
        Normalize over the last dim corresponding to the number of features/channels on that specific layer

    """

    def __init__(self, features) -> None:

        super().__init__()
        self.features = features
        self.norm = nn.LayerNorm([features])

    def forward(self, x):

        x = x.permute(0, 2, 1).contiguous() # (B, C, T) => (B, T, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1).contiguous() # (B, T, C) => (B, C, T)

        return(x)


class CNN_layer(nn.Module):

    """ 
    Convolutional layer: individual conv. layer

    Params
    --------

    filters: number of filters for the conv layer

    kernel: kernel size for the conv layer

    input_channels: number of EEG channels

    Input
    -------
    x: shape(batch size, samples, channels)

    Returns
    --------
    shape(batch size, samples, channels)
    
    """

    def __init__(self, 
        input_channels = 64,
        n_filters = 256, 
        kernel = 8,
        d = .0
    ):
        
        super().__init__()
        self.kernel = kernel

        self.conv = nn.Conv1d(input_channels, n_filters, kernel)
        self.norm = LayerNorm_perm(n_filters)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(d)

        
    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = F.pad(x, (0,self.kernel-1))

        return x
    
class CNN_stack(nn.Module):

    """ 
    Convolutional stack: list all the con layers

    Params
    --------

    filters: int
        number of filters for each layer

    kernel: int
        kernel size for each layer

    input_channels: int
        number of EEG channels

    device

    Input
    -------
    x: shape(batch size, input_channels, samples)

    Returns
    --------
    out: shape(batch size, filters[-1], samples)
    
    """
    
    def __init__(self,
            filters = (256, 256, 256, 128, 128),
            kernels = (8,) * 5,
            input_channels = 64,
            name = 'stack',
            d = 0.2
        ) -> None:

        super().__init__()

        self.kernels = kernels
        self.filters = filters
        self.input_channels = input_channels
        self.name = name

        # First layer with EEG channels as input and the rest of the layers depends on filter size
        self.stack = nn.ModuleList(
            [   CNN_layer(
                    n_filters= filters[0],
                    kernel = kernels[0],
                    input_channels= input_channels,
                    d = d
                )
            ] +
            [
                CNN_layer(
                    n_filters= output,
                    kernel = kernel,
                    input_channels= input,
                    d = d
                )
                for input, output, kernel in zip(filters[:-1], filters[1:], kernels[1:])
            ]
        )

    def forward(self, x):

        for layer in self.stack:
            x = layer(x)

        return x
    
class Out_Ctx_Layer(nn.Module):

    """
    Output context layer: refine the preds based on the prev. predicted samples by padding on the left
    and in consequence apply the filter on the actual sample and the past ones.

    Parameters:
    -----------
    n_filters: int
        Number of filters for the convolutional layer.
    kernel: int
        Kernel size for the convolutional layer.
    input_channels: int
        Number of EEG channels in the input.

    Input
    --------
    x: shape (batch, input_channels, samples)

    Returns
    --------
    shape (batch, input_channels, samples)

    """

    def __init__(self, 
        n_filters = 64, 
        kernel = 32, 
        input_channels = 64,
        name = 'output_ctx',
        d = 0.2
    ):
        super().__init__()

        self.kernel = kernel
        self.n_filters = n_filters
        self.input_channels = input_channels
        self.name = name
        
        self.conv = nn.Conv1d(input_channels, n_filters, kernel)
        self.act = nn.LeakyReLU()
        self.norm = LayerNorm_perm(n_filters)
        self.dropout = nn.Dropout(d)

    def forward(self, x):

        x = F.pad(x, (self.kernel-1, 0))
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x
    
class Linear_Perm(nn.Module):

    """ Linear_Perm: module that perform a linear tranformation permuting over data to perform it over the channel dim
    
    input: shape (batch, channel, samples)
    linear_input: shape (batch, samples, channel)
    out: shape (batch, channel, samples)

    """

    def __init__(self, 
        input_feat, 
        output_feat, 
        bias=True, 
        name = 'linear_perm',
        d = 0.2
    ):
        super().__init__()
        # self.linear = nn.Linear(input_feat, output_feat, bias)
        # Use of convolution insead of pemutations
        self.linear = nn.Conv1d(input_feat, output_feat, kernel_size=1, bias=bias)
        self.name = name
        self.dropout = nn.Dropout(d)

    def forward(self,x):

        # x = x.permute(0, 2, 1).contiguous() # (B, C, T) => (B, T, C)
        x = self.linear(x)
        x = self.dropout(x)
        # x = x.permute(0, 2, 1).contiguous() # (B, T, C) => (B, C, T)

        return x

class VLAAI(nn.Module):

    """Construct the VLAAI model.

    Parameters
    ----------
    n_blocks: int
        Number of repeated blocks to use.
    stack_model: nn.Module
        The extractor model to use.
    output_context_model: nn.Module
        The output context model to use.
    use_skip: bool
        Whether to use skip connections.
    input_channels: int
        Number of EEG channels in the input.
    output_dim: int
        Number of output dimensions.
    name: str
        Name of the model.

    Input
    -------
    x: shape (batch, input_channels, samples)

    Returns
    -------
    x: shape (batch, output_dim, samples)
    
    """
    
    def __init__(self,
        n_blocks = 4,
        stack_model=None,
        output_context_model=None,
        use_skip = True,
        input_channels = 64,
        output_dim = 1,
        dropout = 0.2
    ) -> None:
        
        super().__init__()

        self.use_skip = use_skip
        self.n_blocks = n_blocks

        if stack_model is None:
            stack_model = CNN_stack(input_channels=input_channels, d=dropout)
        if output_context_model is None:
            output_context_model = Out_Ctx_Layer(input_channels=input_channels, n_filters = input_channels, d=dropout)
        
        self.stack_model = stack_model
        self.output_context_model = output_context_model
        self.linear = Linear_Perm(stack_model.filters[-1], input_channels, d=dropout)
        self.out_linear = Linear_Perm(output_context_model.n_filters, output_dim, d=0)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    self.stack_model,
                    self.linear,
                    self.output_context_model
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x, targets=None):

        # Copy the input for the skip connection
        if self.use_skip:
            eeg = torch.clone(x)

        for block in self.blocks:

            x = block(x)

            # Skip connection
            if self.use_skip:
                x = x + eeg 

        preds = self.out_linear(x)
        preds = torch.squeeze(preds, dim=1)
    
        return preds