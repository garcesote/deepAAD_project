import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Configuraciones del modelo
sample_len = 128
channels_num = 64
cnn_kernel_num = 5
cnn_block_len = 4
sa_kq = 50
sa_block_num = 1
sa_channel_dense_num = cnn_kernel_num * sa_block_num
temporal_dropout = 0.5

class MySpaceAttention(nn.Module):
    def __init__(self):
        super(MySpaceAttention, self).__init__()
        self.se_cnn_num = 10
        self.se_pool = 12
        self.my_se_dense = nn.Sequential(
            nn.Conv2d(channels_num, self.se_cnn_num, kernel_size=(1, 1), stride=(1, 1), padding="same"),
            nn.ReLU(),
            nn.Conv2d(self.se_cnn_num, self.se_cnn_num, kernel_size=(1, 1), stride=(1, 1), padding="same"),
            nn.MaxPool2d(kernel_size=(1, self.se_pool)),
            nn.Dropout(temporal_dropout),
            nn.Linear(self.se_cnn_num, 8),
            nn.ReLU(),
            nn.Dropout(temporal_dropout),
            nn.Linear(8, channels_num)
        )

        # Add hooks to monitor shapes
        for layer in self.my_se_dense:
            if isinstance(layer, nn.Module):
                layer.register_forward_hook(self.print_tensor_shape)

    @staticmethod
    def print_tensor_shape(module, input, output):
        print(f"{module.__class__.__name__}: input shape {input[0].shape} -> output shape {output.shape}")


    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B, T // sa_block_num, sa_block_num, C).permute(0, 3, 2, 1)
        w = self.my_se_dense(x)
        w = w.permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 2, 3, 1)
        y = w * x
        y = y.view(B, T, C)
        return y

class MyTemporalAttention(nn.Module):
    def __init__(self):
        super(MyTemporalAttention, self).__init__()
        self.dense_k = nn.Sequential(
            nn.Dropout(temporal_dropout),
            nn.Linear(channels_num, sa_kq),
            nn.ReLU()
        )
        self.dense_q = nn.Sequential(
            nn.Dropout(temporal_dropout),
            nn.Linear(channels_num, sa_kq),
            nn.ReLU()
        )
        self.dense_v = nn.Sequential(
            nn.Dropout(temporal_dropout),
            nn.Linear(channels_num, sa_channel_dense_num),
            nn.Tanh()
        )

    def forward(self, x):
        k = self.dense_k(x)
        q = self.dense_q(x)
        v = self.dense_v(x)
        w = torch.matmul(k, q.transpose(-2, -1)) / math.sqrt(sample_len)
        w = F.softmax(w, dim=-1)
        y = torch.matmul(w, v)
        return y

class EEGAttentionModel(nn.Module):
    def __init__(self):
        super(EEGAttentionModel, self).__init__()
        self.batch_norm = nn.BatchNorm1d(channels_num)
        self.space_attention = MySpaceAttention()
        self.conv1d = nn.Conv1d(channels_num, cnn_kernel_num, kernel_size=5, stride=1, padding=2)
        self.pool1d = nn.MaxPool1d(kernel_size=cnn_block_len)
        self.temporal_attention = MyTemporalAttention()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(sample_len // cnn_block_len * cnn_kernel_num, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.space_attention(x)
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x = self.pool1d(x)
        x = self.temporal_attention(x)
        x = self.fc(x)
        return x