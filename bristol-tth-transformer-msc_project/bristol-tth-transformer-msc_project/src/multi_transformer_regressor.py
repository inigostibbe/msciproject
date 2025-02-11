import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class AnalysisObjectTransformer(L.LightningModule):
    def __init__(
        self,
        embedding=None,
        embed_dim=64,
        output_dim=1,           # classification output dimension
        regression_dim=1,       # regression output dimension (predicting true met)
        expansion_factor=2,
        encoder_layers=3,
        class_layers=3,
        dnn_layers=3,
        num_heads=8,
        hidden_activation=nn.GELU,
        output_activation=nn.Sigmoid,  # or None if you prefer to apply activation in the loss
        dropout=0,
        loss_function=None,
        reco_met_injection=True,  # if True, the reco met token is injected into the sequence
        reco_met_dim=1          # dimension of the reco met input (assumed to be scalar by default)
    ):
        super().__init__()

        # Save basic parameters.
        self.embedding = embedding
        self.regression_dim = regression_dim
        self.class_output_dim = output_dim
        self.reco_met_injection = reco_met_injection

        # Build encoder (self-attention) blocks.
        self.encoder_blocks = nn.ModuleList([
            AttBlock(
                embed_dim=embed_dim,
                expansion_factor=expansion_factor,
                num_heads=num_heads,
                activation=hidden_activation,
                dropout=dropout,
            )
            for _ in range(encoder_layers)
        ])

        # Build class (cross-attention) blocks.
        self.class_blocks = nn.ModuleList([
            AttBlock(
                embed_dim=embed_dim,
                expansion_factor=expansion_factor,
                num_heads=num_heads,
                activation=hidden_activation,
                dropout=dropout,
            )
            for _ in range(class_layers)
        ])

        # Learned classification token.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        # Optionally, add a layer to embed the reco met value.
        if self.reco_met_injection:
            self.reco_met_embedding = nn.Linear(reco_met_dim, embed_dim)

        # Define the DNN head. Its final output dimension is the sum of the classification and regression outputs.
        final_output_dim = self.class_output_dim + self.regression_dim
        layers = []
        for i in range(dnn_layers):
            in_features = embed_dim * expansion_factor if i > 0 else embed_dim
            # For the last layer, output dimension is final_output_dim.
            out_features = (embed_dim * expansion_factor if i < dnn_layers - 1 else final_output_dim)
            layers.append(nn.Linear(in_features, out_features))
            if i < dnn_layers - 1:
                layers.append(hidden_activation())
            else:
                if output_activation is not None:
                    layers.append(output_activation())
            layers.append(nn.Dropout(dropout))
        self.dnn = nn.Sequential(*layers)

        # Loss function for combined classification/regression.
        self.loss_function = loss_function

        # Placeholders for optimizer and scheduler (set externally).
        self.optimizer = None
        self.scheduler_config = None

    def forward(self, x, padding_mask=None, attn_mask=None, reco_met=None):
        # --- Embedding ---
        if self.embedding is not None:
            x = self.embedding(x)

        # Process padding mask: assume nonzero means present; MHA expects True for positions NOT to attend.
        if padding_mask is not None:
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask > 0
            padding_mask = ~padding_mask

        # --- Encoder Blocks (self-attention) ---
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, padding_mask=padding_mask, attn_mask=attn_mask)

        # --- Reco met injection ---
        if self.reco_met_injection and (reco_met is not None):
            # Assume reco_met comes in as (batch,) or (batch, 1). Ensure it has shape (batch, 1).
            if reco_met.dim() == 1:
                reco_met = reco_met.unsqueeze(1)
            # Embed reco met to match the transformer’s embedding dimension.
            reco_met_token = self.reco_met_embedding(reco_met)  # shape: (batch, 1, embed_dim)
            # Append the reco met token to the sequence.
            x = torch.cat([x, reco_met_token], dim=1)
            if padding_mask is not None:
                extra_mask = torch.zeros(padding_mask.size(0), 1, dtype=padding_mask.dtype, device=padding_mask.device)
                padding_mask = torch.cat([padding_mask, extra_mask], dim=1)

        # --- Class Blocks (cross-attention) ---
        # Start with the learned cls token.
        cls_tokens = self.cls_token.expand(x.size(0), 1, -1)  # shape: (batch, 1, embed_dim)
        for class_block in self.class_blocks:
            cls_tokens = class_block(x, x_cls=cls_tokens, padding_mask=padding_mask, attn_mask=attn_mask)
        cls_tokens = cls_tokens.squeeze(1)  # Now shape: (batch, embed_dim)

        # --- DNN Head ---
        out = self.dnn(cls_tokens)  # Final shape: (batch, final_output_dim)
        # Split the outputs into classification and regression parts.
        classification_out = out[..., :self.class_output_dim]
        regression_out = out[..., self.class_output_dim:]
        return classification_out, regression_out

    def predict_step(self, batch, batch_idx):
        # Update your batch to include reco_met and true_met.
        # For example, assume batch = (inputs, labels, weights, mask, event, reco_met, true_met)
        inputs, labels, weights, mask, event, reco_met, true_met = batch
        return self(inputs, padding_mask=mask, reco_met=reco_met)

    def shared_eval(self, batch, batch_idx, suffix):
        # Unpack the batch with the additional reco_met and true_met.
        inputs, labels, weights, mask, event, reco_met, true_met = batch
        classification_out, regression_out = self.forward(inputs, padding_mask=mask, reco_met=reco_met)

        # Compute loss using a custom loss function that expects both outputs.
        # For example, your loss function might be defined to take:
        #   (classification_pred, regression_pred, classification_target, regression_target, event, weights)
        loss_values = self.loss_function(classification_out, regression_out, labels, true_met, event, weights)
        if isinstance(loss_values, dict):
            for loss_name, loss_value in loss_values.items():
                self.log(f"{suffix}/loss_{loss_name}", loss_value, prog_bar=True, on_epoch=True, sync_dist=True)
            assert 'tot' in loss_values.keys()
            return loss_values['tot']
        elif isinstance(loss_values, list):
            loss_value = sum(loss_values)
            self.log(f"{suffix}/loss_tot", loss_value, prog_bar=True, on_epoch=True, sync_dist=True)
            return loss_value
        elif torch.is_tensor(loss_values):
            self.log(f"{suffix}/loss_tot", loss_values, prog_bar=True, on_epoch=True, sync_dist=True)
            return loss_values
        else:
            raise TypeError(f"Loss type {type(loss_values)} not understood")

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler_config(self, scheduler_config):
        self.scheduler_config = scheduler_config

    def configure_optimizers(self):
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set")
        if self.scheduler_config is None:
            return self.optimizer
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler_config}


# --- Example Attention Block (unchanged) ---
class AttBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=2, num_heads=8, activation=nn.GELU, dropout=0):
        super(AttBlock, self).__init__()
        self.num_heads = num_heads
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim * expansion_factor)
        self.activation = activation()
        self.layer_norm4 = nn.LayerNorm(embed_dim * expansion_factor)
        self.linear2 = nn.Linear(embed_dim * expansion_factor, embed_dim)

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        # If a class token is provided, concatenate it to the sequence.
        if x_cls is not None:
            x = torch.cat((x_cls, x), dim=1)  # (batch, seq_len+1, embed_dim)
            if padding_mask is not None:
                padding_mask = torch.concat(
                    (torch.full((padding_mask.shape[0], 1), fill_value=False, device=x.device), padding_mask), dim=1
                )
        x = self.layer_norm1(x)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        if x_cls is not None:
            x_att, _ = self.multihead_attention(x_cls, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)
        else:
            x_att, _ = self.multihead_attention(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)
        x_att = self.layer_norm2(x_att)
        if x_cls is not None:
            x = x_cls + x_att
        else:
            x = x + x_att
        x = self.layer_norm3(x)
        x_linear = self.activation(self.linear1(x))
        x_linear = self.layer_norm4(x_linear)
        x_linear = self.linear2(x_linear)
        x = x + x_linear
        return x
