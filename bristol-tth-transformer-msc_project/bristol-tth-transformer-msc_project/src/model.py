import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


class Embed(nn.Module):
    def __init__(self, input_dim, output_dim, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = F.gelu

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()

        # x: (seq_len, batch, embed_dim)
        x = self.linear(x)
        # Apply activation function
        x = self.activation(x)

        return x

class AttBlock(nn.Module):
    def __init__(self, embed_dims, linear_dims1, linear_dims2, num_heads=8, activation='relu'):
        super(AttBlock, self).__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dims)
        self.multihead_attention = nn.MultiheadAttention(embed_dims, num_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dims)
        self.linear1 = nn.Linear(embed_dims, linear_dims1)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.layer_norm3 = nn.LayerNorm(linear_dims1)
        self.linear2 = nn.Linear(linear_dims1, linear_dims2)

    def forward(self, x, mask=None):
        # Layer normalization 1
        x = self.layer_norm1(x)

        # Multihead Attention
        if mask is not None:
            # Ensure mask has the correct shape for attention
            mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)

        x_att, _ = self.multihead_attention(x, x, x, attn_mask=mask)

        # Skip connection
        x = x + x_att # Skip connection
        # Layer normalization 2
        x = self.layer_norm2(x)
        # Linear layer and activation
        x_linear1 = self.activation(self.linear1(x))
        # Skip connection for the first linear layer
        x = x + x_linear1
        # Layer normalization 3
        x = self.layer_norm3(x_linear1)
        # Linear layer with specified output dimensions
        x_linear2 = self.linear2(x)
        # Skip connection for the second linear layer
        x = x + x_linear2
        return x

class ClassBlock(nn.Module):
    def __init__(self, embed_dims, linear_dims1, linear_dims2, num_heads=8, activation='relu'):
        super(ClassBlock, self).__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dims)
        self.multihead_attention = nn.MultiheadAttention(embed_dims, num_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dims)
        self.linear1 = nn.Linear(embed_dims, linear_dims1)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.layer_norm3 = nn.LayerNorm(linear_dims1)
        self.linear2 = nn.Linear(linear_dims1, linear_dims2)

    def forward(self, x, class_token, mask=None):
        # Concatenate the class token to the input sequence along the sequence length dimension
        x = torch.cat((class_token, x), dim=0)  # (seq_len+1, batch, embed_dim)

        # Layer normalization 1
        x = self.layer_norm1(x)

        # Multihead Attention
        if mask is not None:
            # Ensure mask has the correct shape for attention
            mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)

        x_att, _ = self.multihead_attention(class_token, x, x, attn_mask=mask)

        # Layer normalization 2
        x = self.layer_norm2(x_att)
        x = class_token + x  # Skip connection
        # Linear layer and activation
        x_linear1 = self.activation(self.linear1(x))
        # Layer normalization 3
        x_linear1 = self.layer_norm3(x_linear1)
        # Linear layer with specified output dimensions
        x_linear2 = self.linear2(x_linear1 )
        # Skip connection for the second linear layer
        x = x + x_linear2
        return x


class AnalysisObjectTransformer(nn.Module):
    def __init__(self, input_dim, embed_dims, linear_dims1, linear_dims2, num_heads=8):
        super(AnalysisObjectTransformer, self).__init__()

        # Embedding layer (assumed to be external)
        self.embedding_layer = Embed(input_dim, embed_dims)

        # Three blocks of self-attention
        self.block1 = AttBlock(embed_dims, linear_dims1, linear_dims1, num_heads)
        self.block2 = AttBlock(linear_dims1, linear_dims1, linear_dims1, num_heads)
        self.block3 = AttBlock(linear_dims1, linear_dims2, linear_dims2, num_heads)

        # Class attention layers
        self.cls_token = nn.Parameter(torch.zeros(1, 1, linear_dims2), requires_grad=True)

        self.block5 = ClassBlock(linear_dims2, linear_dims1, linear_dims2, num_heads)
        self.block6 = ClassBlock(linear_dims2, linear_dims1, linear_dims2, num_heads)
        self.block7 = ClassBlock(linear_dims2, linear_dims1, linear_dims2, num_heads)

        # Output linear layer and sigmoid activation
        self.linear_output = nn.Linear(linear_dims2, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):

        # Embedding layer
        x = self.embedding_layer(x)

        # Three blocks of self-attention
        x = self.block1(x, mask)
        x = self.block2(x, mask)
        x = self.block3(x, mask)

        cls_tokens  = self.block5(x, self.cls_token.expand(1, x.size(1), -1), mask)     # (1, N, C)
        cls_tokens  = self.block6(x, cls_tokens, mask)
        cls_tokens  = self.block7(x, cls_tokens, mask)

        # Global average pooling (assuming sequence length is the first dimension)
        x = x.mean(dim=0)

        # Output linear layer and sigmoid activation
        x = self.linear_output(x)
        # x = self.sigmoid(x)

        return x

