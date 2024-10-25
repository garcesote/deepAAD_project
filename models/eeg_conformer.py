import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from dataclasses import dataclass
from utils.functional import correlation
import math

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, config):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.ModuleList([
            nn.Conv2d(1, config.n_embd, (1, config.kernel_temp), (1, 1)), # temporal convolution
            nn.Conv2d(config.n_embd, config.n_embd, (config.kernel_chan, 1), (1, 1)), # channel convolution
            nn.BatchNorm2d(config.n_embd),
            nn.ELU(),
            nn.AvgPool2d((1, config.pool), (1, config.pool_hop)), # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(config.dropout),
        ]
        )

        self.projection = nn.Conv2d(config.n_embd, config.n_embd, (1, 1), stride=(1, 1))  # transpose, conv could enhance fiting ability slightly


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        for layer in self.shallownet:
            x = layer(x)
        x = self.projection(x)
        b, e, c, t = x.shape
        x = x.contiguous().view(b, c*t, e)
        return x
    
class TemporalSelfAttention(nn.Module):

    """ Temporal self-attention module 
    
    Parameters: introduced by config class
    ----------

    config.n_embd: input and output dimension of the embed dimension (number of electrodes with no encoding)
    config.n_heads: number of attention heads (int)
    config.bias: introduce bias to the key, query and value projections (bool)
    config.dropout: dropout applied for the q, k and v and for the output tensor (float)
    config.block_size: number of samples/context for input data (int)

    """

    def __init__(self, config, causal=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.w_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.w_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.w_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.causal = causal
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, q, k, v):
        B, T, C = q.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class MLP(nn.Module):

    """ Multi layer perceptron
    
    Parameters: 
    ------------

    config.n_embd: feature dimension of the data (int: number of electrodes if not encoding), 
    config.bias: introduce bias to the linear projections (bool), 
    config.mlp_ratio: determines the hidden dimension size of the MLP module with respect to the emb_size, 
    config.dropout: dropout probability

    """

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.mlp_ratio * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.mlp_ratio * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class EncoderBlock(nn.Module):

    """ Transformer encoder Block

    Parameters
    -----------

    config.n_embd: embedding dimension
    config.bias: introduce bias to perform layer norm (bool)

    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = TemporalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attn(q=x, k=x, v=x)
        x = self.ln_2(x)
        x = x + self.mlp(x)
        return x

    
class ClassificationHead(nn.Sequential):
    # def __init__(self, emb_size, n_classes):
    def __init__(self, config):
        super().__init__()

        self.dim_tempConv = config.block_size - config.kernel_temp + 1
        self.dim_chanConv = config.eeg_channels - config.kernel_chan + 1
        self.dim_pool = (self.dim_tempConv - config.pool) // config.pool_hop + 1
        self.input_size = self.dim_pool * self.dim_chanConv * config.n_embd
        self.output_dim = 1 if config.unit_output else config.block_size

        if config.classifier:
            self.fc = nn.Sequential(
                nn.Linear(self.input_size, 256),
                nn.ELU(),
                nn.Dropout(config.dropout),
                nn.Linear(256, 32),
                nn.ELU(),
                nn.Dropout(config.dropout),
                nn.Linear(32, self.output_dim)
            )
        else:
            self.fc = nn.Linear(self.input_size, 1)


    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1) # flatten the embed and temp dim
        out = self.fc(x)
        return x, out
    
class TransformerEncoder(nn.Sequential):
    def __init__(self, config):
        super().__init__(*[EncoderBlock(config) for _ in range(config.enc_layers)])
   
@dataclass
class ConformerConfig:
    mlp_ratio: int = 2
    enc_layers: int = 2
    n_head: int = 4
    n_embd: int = 40
    kernel_chan: int = 64
    kernel_temp: int = 8
    pool: int = 20
    pool_hop: int = 4
    eeg_channels: int = 64
    block_size: int = 128 # 2s
    dropout: float = 0.4
    classifier: bool = True
    unit_output: bool = True
    bias: bool = True # True: bias in Linears and LayerNorms. False: a bit better and faster 

class Conformer(nn.Sequential):

    """ EEG Conformer architecture for AAD
    
    This architecture decode the channel and temp dependencies performing a Path Embedding
    to better represent the data and apply a temporal-self-attention transformer
    to model time dependencies on the embedding domain

    """

    def __init__(self, config):
        super().__init__()
        assert config.eeg_channels is not None
        assert config.block_size is not None
        self.config = config

        self.embed = PatchEmbedding(config)
        self.encoder = TransformerEncoder(config)
        self.classif = ClassificationHead(config)

        # report number of parameters
        print("Number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal(module.weight, mode= 'fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, targets=None):

        assert x.shape[1] == self.config.eeg_channels
        
        b, c, t = x.size()

        # Add feature dim
        x = torch.unsqueeze(x, 1)
    
        # Convolutional module to capture temporal and spatial features
        x = self.embed(x)
        
        # Transformer module to capture global temporal dependencies between tokens
        x = self.encoder(x)

        # Classification module to extract the single estimation
        feat_vector, preds = self.classif(x) 

        # Squeeze the output in order to calculate the loss
        preds = torch.squeeze(preds)

        if targets is not None:
            loss = - correlation(preds, targets, batch_dim=self.config.unit_output)
        else:
            loss = None
        return preds, loss