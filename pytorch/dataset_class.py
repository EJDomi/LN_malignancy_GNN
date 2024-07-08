#!/usr/bin/env python
from pathlib import Path
import random

from math import floor
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from scipy.ndimage import center_of_mass
import scipy.io

import SimpleITK as sitk
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import rotate
from skimage.util import random_noise

import torch
from torch_geometric.data import Dataset, Data
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx


class DatasetGeneratorImage(Dataset):
    """
    generate images for pytorch dataset
    """
    def __init__(self, config, resume=None, pre_transform=None):

        self.config = config 
        self.data_path = Path(self.config['data_path'])
        self.patch_path = self.data_path.joinpath(self.config['patch_dir'])
        self.edge_dict = pd.read_pickle(self.data_path.joinpath(f"edge_staging/{self.config['edge_file']}"))
        self.resume = resume
        self.patients = pd.read_pickle(self.data_path.joinpath(self.config['patient_feature_file']))
        # change labels to one class [1,0] -> 0
        self.patients['labels'] = self.patients['labels'].apply(lambda x: list(x)[1]) 

        self.graph_dir = self.data_path.joinpath(f"graph_staging/{self.patch_path.name}_{self.config['edge_file'].replace('.pkl', '')}_{self.config['data_version']}")

        self.rng_rotate = np.random.default_rng(42)
        if self.config['n_rotations'] > 0:
            self.aug_patients = self.patients.copy(deep=True)
            aug_pats = self.patients.copy(deep=True)
            for rot in range(self.config['n_rotations']):
                aug_rot_pats = aug_pats.copy(deep=True)
                aug_rot_pats.index = aug_pats.index.set_levels(aug_pats.index.levels[aug_pats.index.names.index('patients')] + f"_rotation_{rot+1}", level='patients')
                if rot == 0:
                    self.aug_patients = aug_rot_pats
                else:
                    self.aug_patients = pd.concat([self.aug_patients, aug_rot_pats])

            if self.config['balance_classes']:
                n_neg_avg = (aug_pats.groupby('patients').mean('nodes')['labels']<0.5).sum()
                n_pos_avg = (aug_pats.groupby('patients').mean('nodes')['labels']>=0.5).sum()
                ratio_classes = int(floor(n_neg_avg / n_pos_avg))
                for rot in range(ratio_classes):
                    aug_pos_pats = aug_pats.copy(deep=True)[aug_pats['labels']==1]
                    aug_pos_pats.index = aug_pos_pats.index.set_levels(aug_pos_pats.index.levels[aug_pos_pats.index.names.index('patients')] + f"_pos_rotation_{rot+1}", level='patients')
                
            self.aug_patients = self.aug_patients.drop([idx for idx in self.aug_patients.index if 'rotation' not in idx[0]])
        else:
            self.aug_patients = None
        
                            

        super(DatasetGeneratorImage, self).__init__(pre_transform=pre_transform)

    @property
    def raw_paths(self):
        return [f"{self.raw_dir}" for pat in self.patients.index.levels[0]]

    @property
    def raw_dir(self):
        return str(self.patch_path)

    @property
    def processed_dir(self):
        return str(self.graph_dir)

    @property
    def processed_file_names(self):
        if self.config['n_rotations'] > 0:
            file_names = [f"graph_{idx}_{pat}.pt" for idx, pat in enumerate(list(self.patients.index.levels[0])+list(self.aug_patients.index.levels[0]))]
        else:
            file_names = [f"graph_{idx}_{pat}.pt" for idx, pat in enumerate(self.patients.index.levels[0])]
        return file_names


    def download(self):
        pass


    def process(self):
        print("processed graph files not present, starting graph production")
        idx = 0
        for full_pat, group_df in tqdm(self.patients.groupby('patients')):
            if self.resume is not None:
                if idx < self.resume:
                    idx += 1
                    continue
            pat = full_pat.split('_')[0]
            graph_nx = self.edge_dict[pat]

            if 'rotation' in full_pat:
                angle = self.rng_rotate.integers(-30, high=30)

            for node in group_df.index:
                node = node[1] #index is a multiindex, with node # in second position

                node_name = f"{pat}_{node}"

                patch = scipy.io.loadmat(self.patch_path.joinpath(f"ct/{node_name}.mat"))['roi_patch_ct']
                seg = scipy.io.loadmat(self.patch_path.joinpath(f"seg/{node_name}.mat"))['roi_patch_seg']

                if 'rotation' in full_pat:
                    patch = self.apply_rotation(patch, angle)
                    seg = self.apply_rotation(seg, angle)

                graph_nx.nodes[node]['x'] = torch.from_numpy(np.expand_dims(patch, 0))
                #graph_nx.nodes[node]['x'] = torch.from_numpy(np.stack((patch, seg)))

                # input y will be malignancy status, y1 will be the mask for segmentation guided network
                graph_nx.nodes[node]['y1'] = torch.from_numpy(np.expand_dims(seg, 0))
                graph_nx.nodes[node]['y'] = group_df.loc[(full_pat, node), 'labels']
                graph_nx.nodes[node]['features'] = torch.from_numpy(group_df.loc[(full_pat, node), ['dist_to_pri', 'pri_location', 'pri_lat', 'ln_level', 'ln_lat']].values)
                graph_nx.graph['patient'] = full_pat
              
            graph_pyg = from_networkx(graph_nx)
           
            torch.save(graph_pyg, f"{self.processed_dir}/graph_{idx}_{full_pat}.pt")
            idx += 1
      

        if self.aug_patients is not None: 
            for full_pat, group_df in tqdm(self.aug_patients.groupby('patients')):
                if self.resume is not None:
                    if idx < self.resume:
                        idx += 1
                        continue
                pat = full_pat.split('_')[0]
                graph_nx = self.edge_dict[pat]

                if 'rotation' in full_pat:
                    angle = self.rng_rotate.integers(-30, high=30)

                for node in group_df.index:
                    node = node[1] #index is a multiindex, with node # in second position

                    node_name = f"{pat}_{node}"

                    patch = scipy.io.loadmat(self.patch_path.joinpath(f"ct/{node_name}.mat"))['roi_patch_ct']
                    seg = scipy.io.loadmat(self.patch_path.joinpath(f"seg/{node_name}.mat"))['roi_patch_seg']

                    if 'rotation' in full_pat:
                        patch = self.apply_rotation(patch, angle)
                        seg = self.apply_rotation(seg, angle)

                    graph_nx.nodes[node]['x'] = torch.from_numpy(np.expand_dims(patch, 0))
                    #graph_nx.nodes[node]['x'] = torch.from_numpy(np.stack((patch, seg)))

                    # input y will be malignancy status, y1 will be the mask for segmentation guided network
                    graph_nx.nodes[node]['y1'] = torch.from_numpy(seg)
                    graph_nx.nodes[node]['y'] = group_df.loc[(full_pat, node), 'labels']
                    graph_nx.nodes[node]['features'] = torch.from_numpy(group_df.loc[(full_pat, node), ['dist_to_pri', 'pri_location', 'pri_lat', 'ln_level', 'ln_lat']].values)
                  


                graph_pyg = from_networkx(graph_nx)
                

                torch.save(graph_pyg, f"{self.processed_dir}/graph_{idx}_{full_pat}.pt")
                idx += 1
 

    def len(self):
        return len(self.patients)


    def get(self, idx):
        if self.config['n_rotations'] > 0:
            pat = pd.concat([self.patients, self.aug_patients]).index.get_level_values(0).unique()[idx]
        else:
            pat = self.patients.index.get_level_values(0).unique()[idx]
        data = torch.load(f"{self.processed_dir}/graph_{idx}_{pat}.pt")
        #data = torch.load(f"{self.processed_dir}/graph_{idx}.pt")
        return data


    def apply_rotation(self, arr, angle):
        arr = rotate(arr, angle, preserve_range=True)
        return arr



