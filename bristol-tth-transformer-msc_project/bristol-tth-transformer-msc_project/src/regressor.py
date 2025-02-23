import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import lightning as L


class Embedding(nn.Module):
    def __init__(self, input_dim, embed_dims, normalize_input=True, activation=nn.GELU):
        # input_dim = train_X.shape[-1],   # Which is 6 as shape = (batch, 10, 6)
        # embed_dims = [64], 

        super().__init__()
        if not isinstance(embed_dims,(tuple,list)):
            embed_dims = [embed_dims]
        assert len(embed_dims) >= 1

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None

        layers = [
            nn.Linear(input_dim,embed_dims[0]),
            activation(),
        ]
        for dim_in,dim_out in zip(embed_dims[:-1],embed_dims[1:]):
            layers.append(nn.Linear(dim_in,dim_out))
            layers.append(activation()) 
        self.layers = nn.Sequential(*layers)

        self.dim = embed_dims[-1]

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, seq_len, embed_dim)
            # batch norm expects (batch, embed_dim, sequence length)
            x = self.input_bn(x.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()

        return self.layers(x)

class AttBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=2, num_heads=8, activation=nn.GELU, dropout=0):
        super(AttBlock, self).__init__()

        self.num_heads = num_heads
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            dropout = dropout,
            batch_first = True,
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim * expansion_factor)
        self.activation = activation()
        self.layer_norm4 = nn.LayerNorm(embed_dim * expansion_factor)
        self.linear2 = nn.Linear(embed_dim * expansion_factor, embed_dim)

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        # Class token provided -> concat to do the class attention #
        if x_cls is not None:
            x = torch.cat((x_cls, x), dim=1)  # (batch, seq_len+1, embed_dim)
            if padding_mask is not None:
                # Need to add mask=False for cls token
                padding_mask = torch.concat(
                    (
                        torch.full((padding_mask.shape[0],1),fill_value=False).to(x.device),
                        padding_mask,
                    ),
                    dim = 1,
                )
        # Layer normalization 1
        x = self.layer_norm1(x)
        # Multihead Attention
        if attn_mask is not None:
            # Ensure mask has the correct shape for attention
            # Repeat for each head
            attn_mask = attn_mask.repeat(self.num_heads,1,1)
        if x_cls is not None:
            x_att, _ = self.multihead_attention(x_cls, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)
        else:
            x_att, _ = self.multihead_attention(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)
        # Layer normalization 2
        x_att = self.layer_norm2(x_att)
        # Skip connection
        if x_cls is not None:
            x = x_cls + x_att # Skip connection
        else:
            x = x + x_att
        # Layer normalization 3
        x = self.layer_norm3(x)
        # Linear layer 1 #
        x_linear = self.activation(self.linear1(x))
        # Layer normalization 4
        x_linear = self.layer_norm4(x_linear)
        # Linear layer 2
        x_linear = self.linear2(x_linear)
        # Final skip connection #
        x = x + x_linear
        return x

class AnalysisObjectTransformer(L.LightningModule):
    def __init__(self, embedding=None, embed_dim=64, output_dim=1, expansion_factor=2, encoder_layers=3, class_layers=3, dnn_layers=3, num_heads=8, hidden_activation=nn.GELU, output_activation=None, dropout=0, loss_function=None):
        super().__init__()

        # Embedding layer (assumed to be external)
        self.embedding = embedding
 
        # Three blocks of self-attention
        self.encoder_blocks = nn.ModuleList(
            [
                AttBlock(
                    embed_dim = embed_dim,
                    expansion_factor = expansion_factor,
                    num_heads = num_heads,
                    activation = hidden_activation,
                )
                for _ in range(encoder_layers)
            ]
        )

        # Output linear layer and sigmoid activation
        layers = []
        for i in range(dnn_layers):
            layers.append(
                nn.Linear(
                    (embed_dim+1) * expansion_factor if i > 0 else (embed_dim+1),  # Adding one to the embed_dim for the reco_met that is appeneded after attention
                    (embed_dim+1) * expansion_factor if i < dnn_layers-1 else 1,
                )
            )
            if i < dnn_layers-1:
                layers.append(hidden_activation())
            layers.append(nn.Dropout(dropout))
        self.dnn = nn.Sequential(*layers)

        # Loss function #
        self.loss_function = loss_function # This is the classification loss

    def forward(self, x, padding_mask=None, attn_mask=None, reco_met=None, phi=None):

        # Embedding layer
        if self.embedding is not None:
            x = self.embedding(x)

        # Process masks #
        if padding_mask is not None:
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask > 0
                # 1/True : jet is present, while 0/False : jet is absent
            padding_mask = ~padding_mask
            # from the doc in MHA : True = not attended
            # so we reverse our mask
        if attn_mask is not None:
            raise NotImplementedError

        # Applying the attention blocks #
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x,padding_mask=padding_mask,attn_mask=attn_mask)

        # Final dnn #
        x = x.mean(dim=1)

        if reco_met is not None:
            # Here we can append the InputMet and InputPhi
            x = torch.cat([x, reco_met], dim=1)
        else:
            # Stop training and raise an error
            raise ValueError('reco_met is None') # This is to ensure that the dimensions are correct (leads to other issues)

        x = self.dnn(x)

        return x

    def predict_step(self, batch, batch_idx):
        inputs, labels, weights, mask, event, phi, target = batch
        outputs = self.forward(inputs,padding_mask=mask, reco_met=event) # Self.forward instead of self() for clarity
        return outputs
    

    def shared_eval(self, batch, batch_idx, suffix):
        inputs, labels, weights, mask, event, phi, target = batch

        outputs = self.forward(inputs,padding_mask=mask, reco_met=event) # This was self() before but now self.forward() for clarity

        assert self.loss_function is not None

        loss_values = self.loss_function(outputs,target)

        if isinstance(loss_values,dict):
            for loss_name,loss_value in loss_values.items():
                self.log(f"{suffix}/loss_{loss_name}", loss_value, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True) # CHANGED TO LOG ON EPOCH and sync true
            assert 'tot' in loss_values.keys()
            return loss_values['tot']
        elif isinstance(loss_values,list):
            loss_value = sum(loss_values)
            self.log(f"{suffix}/loss_tot",loss_value, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return sum(loss_values)
        elif torch.is_tensor(loss_values):
            self.log(f"{suffix}/loss_tot",loss_values, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return loss_values
        else:
            raise TypeError(f'Type {type(loss_values)} of loss not understood')

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch,batch_idx,'train')    

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch,batch_idx,'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch,batch_idx,'test')

    def set_optimizer(self,optimizer):
        self.optimizer = optimizer

    def set_scheduler_config(self,scheduler_config):
        self.scheduler_config = scheduler_config

    def configure_optimizers(self):
        if self.optimizer is None:
            raise RuntimeError('Optimizer not set')
        if self.scheduler_config is None:
            return self.optimizer
        else:
            return {
                'optimizer' : self.optimizer,
                'lr_scheduler': self.scheduler_config,
            }
