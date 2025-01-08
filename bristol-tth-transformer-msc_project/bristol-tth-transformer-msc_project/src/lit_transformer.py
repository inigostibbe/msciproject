import torch.nn.functional as F
import lightning as L

from losses import distance_corr


class LitTransformer(L.LightningModule):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = F.binary_cross_entropy_with_logits
        
    def predict_step(self, batch, batch_idx):
        inputs, labels, weights, mask, event = batch
        outputs = self.model(inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        inputs, labels, weights, mask, event = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels, weight=weights.unsqueeze(-1)) + 0.15 * distance_corr(outputs, event, weights)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels, weights, mask, event = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels, weight=weights.unsqueeze(-1)) + 0.15 * distance_corr(outputs, event, weights)
        self.log("val_loss", loss)
        
    def test_step(self, batch, batch_idx):
        inputs, labels, weights, mask, event = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels, weight=weights.unsqueeze(-1))
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer
    
    def on_epoch_end(self):
        # Optionally log metrics at the end of each epoch
        train_loss = self.trainer.callback_metrics['train_loss']
        val_loss = self.trainer.callback_metrics['val_loss']
        self.log('epoch_train_loss', train_loss)
        self.log('epoch_val_loss', val_loss)