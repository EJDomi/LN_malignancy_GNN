import os
from pathlib import Path
import copy
from datetime import datetime
from collections import OrderedDict
import time
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from monai.data import partition_dataset_classes, partition_dataset

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchmetrics

import pytorch_lightning as L

from LN_malignancy_GNN.pytorch.lightning_GNN import CNN_GNN
from LN_malignancy_GNN.pytorch.dataset_class import DatasetGeneratorImage

#models that use edge_attr

class RunModel(object):
    def __init__(self, config):
        self.config = config
        L.seed_everything(self.config['seed'])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_weights = None

        if self.config['log_dir'] is None:
            self.log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            self.log_dir = os.path.join('logs', self.config['log_dir'])

        self.metric_dir = os.path.join(self.log_dir, 'metric_dfs')
        Path(self.metric_dir).mkdir(parents=True, exist_ok=True)
        print(f"logs are located at: {self.log_dir}")
        #self.writer = SummaryWriter(self.log_dir)
        print('remember to set the data')


    def set_model(self):
        """
        sets and assigns GNN model to self.model
        make sure to assign dataset before calling set_model
        """
        self.model = CNN_GNN(self.config) 


    def set_data(self, resume=None):

        if self.config['dataset_name'] == 'UTSW':
            self.data = DatasetGeneratorImage(config=self.config, resume=resume)


    def set_train_test_split(self):

        self.folds = None
        if self.config['preset_folds']:
            self.folds = pd.read_pickle(self.data.data_path.joinpath(self.config['preset_fold_file']))
        else:
            self.folds = partition_dataset_classes(range(len(self.data.patients.index.get_level_values(0).unique())), self.data.patients.groupby('patients').mean('nodes')['labels']>=0.5, num_partitions=5, shuffle=True, seed=self.config['seed'])
            with open(self.data.data_path.joinpath(self.config['preset_fold_file']), 'wb') as f:
                pickle.dump(self.folds, f)
                f.close()

        self.train_folds = [[0,1,2],
                            [4,0,1],
                            [3,4,0],
                            [2,3,4],
                            [1,2,3]]
        self.val_folds = [3, 2, 1, 0, 4]
        self.test_folds = [4, 3, 2, 1, 0]

        self.nested_train_folds = [[[0,1,2],[1,2,3],[2,3,0],[3,0,1]],
                                   [[4,0,1],[0,1,2],[1,2,4],[2,4,0]],
                                   [[3,4,0],[4,0,1],[0,1,3],[1,3,4]],
                                   [[2,3,4],[3,4,0],[4,0,2],[0,2,3]],
                                   [[1,2,3],[2,3,4],[3,4,1],[4,1,2]]]
        self.nested_val_folds = [[3,0,1,2],
                                 [2,4,0,1],
                                 [1,3,4,0],
                                 [0,2,3,4],
                                 [4,1,2,3]]

        self.train_splits = [self.folds[i]+self.folds[j]+self.folds[k] for i,j,k in self.train_folds]
        self.val_splits = [self.folds[i] for i in self.val_folds]
        self.test_splits = [self.folds[i] for i in self.test_folds]

        self.aug_splits = []
        if self.config['n_rotations'] > 0:
           for fold_idx, fold in enumerate(self.train_splits):
               aug_pats = self.data.aug_patients.index.get_level_values(0).unique()
               fold_pats = self.data.patients.iloc[fold].index.get_level_values(0).unique()
               aug_fold_pats = [pat for pat in aug_pats if np.any([(fold_pat[0] in pat) for fold_pat in fold_pats])]
               aug_fold_idx = self.data.patients.index.get_indexer(aug_fold_pats) + len(self.data.patients.index.get_level_values(0).unique())
               self.train_splits[fold_idx].extend(aug_fold_idx)

        self.class_weights = [3,3,3,3,3]
        #for split in self.train_splits:
        #    self.class_weights.append([len(self.data.patients['labels'].values[split][self.data.patients.values[split]['labels']==0]) / np.sum(self.data.y.values[split]['labels'])])

        self.nested_train_splits = [[self.folds[i]+self.folds[j]+self.folds[k] for i,j,k in nest] for nest in self.nested_train_folds]
        self.nested_val_splits = [[self.folds[i] for i in nest] for nest in self.nested_val_folds] 

            
    def set_data_module(self):
        self.data_module_cross_val = [LightningDataset(train_dataset=self.data[fold], val_dataset=self.data[self.val_splits[idx]], test_dataset=self.data[self.test_splits[idx]], batch_size=self.config['batch_size'], num_workers=19, pin_memory=True, persistent_workers=False, shuffle=True) for idx, fold in enumerate(self.train_splits)] 


    def set_callbacks(self):
        self.callbacks = []

        #Checkpoint options
        self.callbacks.append(L.callbacks.ModelCheckpoint(
            monitor='val_loss', 
            mode='min', 
            save_top_k=5,
            dirpath=os.path.join(self.log_dir, 'top_models'),
            filename='model_{epoch:02d}_{val_loss:.2f}_{val_auc:.2f}',
            ))
     
        #self.callbacks.append(L.callbacks.EarlyStopping(monitor='val_loss', patience=10, check_on_train_epoch_end=False))

        self.callbacks.append(L.callbacks.LearningRateFinder(min_lr=1e-6, max_lr=1e-1))

               
    def run(self, resume=False, resume_idx=None):
        self.trainers = []
        self.metrics = {}
        
        self.metrics['val'] = []
        self.metrics['test'] = []

        for idx in range(5):
            self.set_model()
            self.set_callbacks()

            self.trainers.append(L.Trainer(
                max_epochs=self.config['n_epochs'],
                accelerator="auto",
                devices=self.config['gpu_device'] if torch.cuda.is_available() else None,
                logger=L.loggers.CSVLogger(save_dir=self.log_dir),
                callbacks=self.callbacks,
                #check_val_every_n_epoch = 1,
                #auto_lr_find=True
                ))
 
            self.trainers[idx].fit(self.model, datamodule=self.data_module_cross_val[idx])
            self.metrics['val'].append(self.trainers[idx].validate(self.model, datamodule=self.data_module_cross_val[idx]))
            self.metrics['test'].append(self.trainers[idx].test(self.model, datamodule=self.data_module_cross_val[idx]))

        pd.DataFrame(self.metrics['val']).to_pickle(os.path.join(self.metric_dir, 'val_metrics.pkl'))
        pd.DataFrame(self.metrics['test']).to_pickle(os.path.join(self.metric_dir, 'test_metrics.pkl'))




    def get_predictions(self):
        """
        get predictions from list of trainers stored in self.trainers
        requires run() to be executed as a prerequisite
        this will get a set of predictions for each fold
        """

        test_predictions_dict = []
        val_predictions_dict = []
        for idx, trainer in enumerate(self.trainers):
            test_predictions_dict.append({})
            val_predictions_dict.append({})
            tmp_test_targets = []
            tmp_val_targets = []
            test_predictions_dict[idx]['predictions'] = torch.cat(trainer.predict(trainer.model, self.data_module_cross_val[idx].test_dataloader()))
            val_predictions_dict[idx]['predictions'] = torch.cat(trainer.predict(trainer.model, self.data_module_cross_val[idx].val_dataloader()))
            for batch in self.data_module_cross_val[idx].test_dataloader():
                tmp_test_targets.append(batch.y)
            for batch in self.data_module_cross_val[idx].val_dataloader():
                tmp_val_targets.append(batch.y)

            test_predictions_dict[idx]['targets'] = torch.cat(tmp_test_targets)
            val_predictions_dict[idx]['targets'] = torch.cat(tmp_val_targets)

        self.test_predictions_df = pd.DataFrame(test_predictions_dict)
        self.val_predictions_df = pd.DataFrame(val_predictions_dict)

        self.test_predictions_df.to_pickle(os.path.join(self.metric_dir, 'test_predictions.pkl'))
        self.val_predictions_df.to_pickle(os.path.join(self.metric_dir, 'val_predictions.pkl'))


