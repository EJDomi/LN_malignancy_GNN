import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import torchvision
import torchmetrics

import pytorch_lightning as L

import LN_malignancy_GNN.pytorch.user_metrics as um
import LN_malignancy_GNN.pytorch.extractor_networks as en
import LN_malignancy_GNN.pytorch.gnn_networks as graphs


class Classify(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.classify = nn.Linear(in_channels, n_classes)
        #self.classify = nn.LazyLinear(n_classes)


    def forward(self, x, clinical=None):
        if clinical is not None:
            x = torch.cat((x, clinical), -1)
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
        self.extractor = getattr(en, self.config['extractor_name'])(n_classes=self.config['extractor_channels'], in_channels=self.config['n_in_channels'], dropout=self.config['dropout'])
        #self.gnn = getattr(graphs, self.config['model_name'])(self.config['extractor_channels'], hidden_channels=self.config['n_hidden_channels'], n_classes=self.config['n_hidden_channels'], edge_dim=self.config['edge_dim'], dropout=self.config['dropout'], num_layers=self.config['num_deep_layers'])
        
        self.gnn = getattr(graphs, self.config['model_name'])(self.config['extractor_channels'], hidden_channels=self.config['n_hidden_channels'], n_classes=self.config['gnn_out_channels'], edge_dim=self.config['edge_dim'], dropout=self.config['dropout'])
 
        self.classify = Classify(in_channels=self.config['gnn_out_channels']+(self.config['n_clinical'] if self.config['use_clinical'] else 0), n_classes=self.config['n_classes'])

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.config['class_weight']]))
        #self.val_loss_fn = nn.BCEWithLogitsLoss()
        #self.test_loss_fn = nn.BCEWithLogitsLoss()
        self.m_fn = um.MMetric(0.6, 0.4)
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

        #for m in self.extractor.modules():
        #    self.init_params(m)
        for m in self.gnn.modules():
            self.init_params(m)
        for m in self.classify.modules():
            self.init_params(m)
            #if isinstance(m, nn.BatchNorm3d):
            #    m.weight.requires_grad = False

        for param in self.extractor.parameters():
            param.data = param.data.float()
        for param in self.gnn.parameters():
            param.data = param.data.float()
        for param in self.classify.parameters():
            param.data = param.data.float()
        self.save_hyperparameters()


    def init_params(self, m):
        """
           Following is the doc string from stolen function:
                Initialize the parameters of a module.
                Parameters
                ----------
                m
                    The module to initialize.
                Notes
                -----
                Convolutional layer weights are initialized from a normal distribution
                as described in [1]_ in `fan_in` mode. The final layer bias is
                initialized so that the expected predicted probability accounts for
                the class imbalance at initialization.
                References
                ----------
                .. [1] K. He et al. ‘Delving Deep into Rectifiers: Surpassing
                   Human-Level Performance on ImageNet Classification’,
                   arXiv:1502.01852 [cs], Feb. 2015.
        """
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)


    def normalize_features(self, batch):
        '''
        function to retrieve the difference in distance to primary and LN laterality
        to be used as edge features in gnn
        '''
        features = batch.features.float()
        batch_mean = batch.features[:, 0].mean()
        batch_std = batch.features[:, 0].std()
        if len(features[:, 0]) == 1:
            features[:, 0] = 0.
        else:
            features[:, 0] = (batch.features[:,0] - batch_mean) / batch_std
        return features


    def _shared_eval_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index

        if self.config['n_in_channels'] == 1 and self.config['region'] == 'MASK':
            x = torch.where(batch.y1, x, 0.)
        elif self.config['n_in_channels'] == 1 and self.config['region'] == 'ROI':
            x = x 
        elif self.config['n_in_channels'] == 2 and self.config['region'] == 'ROI':
            x = torch.cat((x, x), dim=1)
        elif self.config['n_in_channels'] == 2:
            x = torch.cat((x, batch.y1), dim=1)

        if self.config['use_clinical']:
            features = self.normalize_features(batch)
        else:
            features = None
        if self.config['edge_dim'] is not None:
            edge_attr = torch.nan_to_num(batch.edge_attr, nan=0.0)
        else:
            edge_attr = None

        if 'vit' in self.config['extractor_name']:
            x = self.extractor(x)[0]
            #x = self.avg_pool(x)
        else:
            x = self.extractor(x)

        if x.dim() == 1:
            x = x.squeeze().unsqueeze(0)

        x = self.gnn(x=x, edge_index=edge_index, batch=batch.batch, edge_attr=edge_attr) 
 
      
        if self.config['include_primary']:
            patch_type = np.squeeze(batch.patch_type)
            x = x[patch_type != 'primary']
            if features is not None:
                features = features[patch_type != 'primary']

        return self.classify(x, features)
        

    def training_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index

        if self.config['n_in_channels'] == 1 and self.config['region'] == 'MASK':
            x = torch.where(batch.y1, x, 0.)
        elif self.config['n_in_channels'] == 1 and self.config['region'] == 'ROI':
            x = x 
        elif self.config['n_in_channels'] == 2 and self.config['region'] == 'ROI':
            x = torch.cat((x, x), dim=1)
        elif self.config['n_in_channels'] == 2:
            x = torch.cat((x, batch.y1), dim=1)

        if self.config['use_clinical']:
            features = self.normalize_features(batch)
        else:
            features = None

        if self.config['freeze_extractor']:
            with torch.no_grad():
                x = self.extractor(x)
        else:
            if 'vit' in self.config['extractor_name']:
                x = self.extractor(x)[0]
                #x = self.avg_pool(x)
            else:
                x = self.extractor(x)
        if self.config['edge_dim'] is not None:
            edge_attr = torch.nan_to_num(batch.edge_attr, nan=0.0)
        else:
            edge_attr = None

        if x.dim() == 1:
            x = x.squeeze().unsqueeze(0)

        x = self.gnn(x=x, edge_index=edge_index, batch=batch.batch, edge_attr=edge_attr)  
       
        if self.config['include_primary']:
            patch_type = np.squeeze(batch.patch_type)
            x = x[patch_type != 'primary']
            if features is not None:
                features = features[patch_type != 'primary']
            y = batch.y[patch_type != 'primary']
        else:
            y = batch.y
        pred = self.classify(x, features) 

        if pred.dim() > 2:
            pred = pred.squeeze()
        if y.dim() > 2:
            y = y.squeeze()

        loss = self.loss_fn(pred, y.to(torch.float))

        self.auc_fn(pred, y) 
        self.ap_fn(pred, y.to(torch.int64)) 
        self.sen_fn(pred, y) 
        self.spe_fn(pred, y) 

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(batch.batch), prog_bar=True)
        self.log("train_auc", self.auc_fn, on_step=False, on_epoch=True, batch_size=len(batch.batch), prog_bar=True)
        self.log("train_ap", self.ap_fn, on_step=False, on_epoch=True, batch_size=len(batch.batch))
        self.log("train_m", self.m_fn(self.sen_fn.compute(), self.spe_fn.compute()), on_step=False, on_epoch=True, batch_size=len(batch.batch))
        self.log("train_sen", self.sen_fn, on_step=False, on_epoch=True, batch_size=len(batch.batch))
        self.log("train_spe", self.spe_fn, on_step=False, on_epoch=True, batch_size=len(batch.batch))

        return loss


    def validation_step(self, batch, batch_idx):
        pred = self._shared_eval_step(batch, batch_idx)

        if self.config['include_primary']:
            patch_type = np.squeeze(batch.patch_type)
            y = batch.y[patch_type != 'primary']
        else:
            y = batch.y

        if pred.dim() > 2:
            pred = pred.squeeze()
        if y.dim() > 2:
            y = y.squeeze()

        val_loss = self.loss_fn(pred, y.to(torch.float))

        self.val_auc_fn(pred, y) 
        self.val_ap_fn(pred, y.to(torch.int64)) 
        self.val_sen_fn(pred, y) 
        self.val_spe_fn(pred, y) 

        self.log_dict({"val_loss": torch.tensor([val_loss]),
        "val_auc": self.val_auc_fn,
        "val_ap": self.val_ap_fn,
        "val_m": self.m_fn(self.val_sen_fn.compute(), self.val_spe_fn.compute()),
        "val_sen": self.val_sen_fn,
        "val_spe": self.val_spe_fn,
        }, batch_size=len(batch.batch), prog_bar=True)

        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        pred = self._shared_eval_step(batch, batch_idx)

        if self.config['include_primary']:
            patch_type = np.squeeze(batch.patch_type)
            y = batch.y[patch_type != 'primary']
        else:
            y = batch.y

        if pred.dim() > 2:
            pred = pred.squeeze()
        if y.dim() > 2:
            y = y.squeeze()

        test_loss = self.loss_fn(pred, y.to(torch.float))

        self.test_auc_fn(pred, y) 
        self.test_ap_fn(pred, y.to(torch.int64)) 
        self.test_sen_fn(pred, y) 
        self.test_spe_fn(pred, y) 

        self.log("test_auc", self.test_auc_fn)
        self.log("test_ap", self.test_ap_fn)
        self.log("test_m", self.m_fn(self.test_sen_fn.compute(), self.test_spe_fn.compute()))
        self.log("test_sen", self.test_sen_fn)
        self.log("test_spe", self.test_spe_fn)

    def predict_step(self, batch, batch_idx):
        x = self._shared_eval_step(batch, batch_idx)

        if x.dim() > 2:
            x = x.squeeze()

        turn = nn.Sigmoid()
        pred = turn(x)
        return pred


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        
        #lr_scheduler_config = {
        #    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        #    "interval": "epoch",
        #    "frequency": self.config['lr_patience'],
        #    "monitor": "val_loss",
        #    "strict": True,
        #    "name": None,
        #}

        #return {'optimizer': optimizer,
        #        'lr_scheduler': lr_scheduler_config,}
        return {'optimizer': optimizer,}

