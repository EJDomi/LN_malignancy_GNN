import torch
from torch import nn
import torch.nn.functional as F

import torchvision
import torchmetrics

import pytorch_lightning as L

import LN_malignancy_GNN.pytorch.extractor_networks as en
import LN_malignancy_GNN.pytorch.gnn_networks as graphs


class Classify(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.classify = nn.Linear(in_channels, n_classes)
        #self.classify = nn.LazyLinear(n_classes)


    def forward(self, x, clinical=None):
        if clinical is not None:
            x = torch.cat((x, clinical), 1)
            x = self.classify(x) 
        else:
            x = self.classify(x)

        if len(x) == 1:
            x = x.squeeze().unsqueeze(0)
        else:
            x = x.squeeze()
        return x


class CNN_GNN(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learning_rate = self.config['learning_rate']
        self.extractor = getattr(en, self.config['extractor_name'])(in_channels=self.config['n_in_channels'], dropout=self.config['dropout'])
        self.gnn = getattr(graphs, self.config['model_name'])(self.config['extractor_channels'], hidden_channels=self.config['n_hidden_channels'], n_classes=self.config['n_hidden_channels'], edge_dim=self.config['edge_dim'], dropout=self.config['dropout'])
 
        self.classify = Classify(in_channels=self.config['n_hidden_channels'], n_classes=self.config['n_classes'])

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.val_loss_fn = nn.BCEWithLogitsLoss()
        self.test_loss_fn = nn.BCEWithLogitsLoss()

        self.auc_fn = torchmetrics.classification.BinaryAUROC()
        self.ap_fn = torchmetrics.classification.BinaryAveragePrecision()
        self.spe_fn = torchmetrics.classification.BinarySpecificity()
        self.sen_fn = torchmetrics.classification.BinaryRecall()

        self.val_auc_fn = torchmetrics.classification.BinaryAUROC()
        self.val_ap_fn = torchmetrics.classification.BinaryAveragePrecision()
        self.val_spe_fn = torchmetrics.classification.BinarySpecificity()
        self.val_sen_fn = torchmetrics.classification.BinaryRecall()

        self.test_auc_fn = torchmetrics.classification.BinaryAUROC()
        self.test_ap_fn = torchmetrics.classification.BinaryAveragePrecision()
        self.test_spe_fn = torchmetrics.classification.BinarySpecificity()
        self.test_sen_fn = torchmetrics.classification.BinaryRecall()

        self.save_hyperparameters()


    def forward(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index

        if batch.edge_attr is not None:
            edge_attr = batch.edge_attr
        else:
            edge_attr = None

        x = self.extractor(x)
        x = self.gnn(x=x, edge_index=edge_index, batch=batch.batch, edge_attr=edge_attr)  

        return self.classify(x)
        

    def training_step(self, batch, batch_idx):
        
        pred = self.forward(batch, batch_idx) 

        loss = self.loss_fn(pred, batch.y.to(torch.float))

        self.auc_fn(pred, batch.y) 
        self.ap_fn(pred, batch.y.to(torch.int64)) 
        self.sen_fn(pred, batch.y) 
        self.spe_fn(pred, batch.y) 

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(batch.batch))
        self.log("train_auc", self.auc_fn, on_step=False, on_epoch=True, batch_size=len(batch.batch))
        self.log("train_ap", self.ap_fn, on_step=False, on_epoch=True, batch_size=len(batch.batch))
        self.log("train_sen", self.sen_fn, on_step=False, on_epoch=True, batch_size=len(batch.batch))
        self.log("train_spe", self.spe_fn, on_step=False, on_epoch=True, batch_size=len(batch.batch))

        return loss


    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch, batch_idx)

        val_loss = self.val_loss_fn(pred, batch.y.to(torch.float))

        self.val_auc_fn(pred, batch.y) 
        self.val_ap_fn(pred, batch.y.to(torch.int64)) 
        self.val_sen_fn(pred, batch.y) 
        self.val_spe_fn(pred, batch.y) 

        self.log_dict({"val_loss": val_loss,
        "val_auc": self.val_auc_fn,
        "val_ap": self.val_ap_fn,
        "val_sen": self.val_sen_fn,
        "val_spe": self.val_spe_fn,
        }, batch_size=len(batch.batch))

        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        pred = self.forward(batch, batch_idx)

        test_loss = self.test_loss_fn(pred, batch.y.to(torch.float))

        self.test_auc_fn(pred, batch.y) 
        self.test_ap_fn(pred, batch.y.to(torch.int64)) 
        self.test_sen_fn(pred, batch.y) 
        self.test_spe_fn(pred, batch.y) 

        self.log("test_auc", self.test_auc_fn)
        self.log("test_ap", self.test_ap_fn)
        self.log("test_sen", self.test_sen_fn)
        self.log("test_spe", self.test_spe_fn)

    def predict_step(self, batch, batch_idx):
        x = self.forward(batch, batch_idx)
        turn = nn.Sigmoid()
        pred = turn(x)
        return pred


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            "interval": "epoch",
            "frequency": self.config['lr_patience'],
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }

        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler_config,}

