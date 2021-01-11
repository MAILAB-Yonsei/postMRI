"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random
import h5py
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm

# You should build your custom dataset as below.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, challenge='train', sample_rate=1, cv='1'):
        
        # TODO
        # 1. Initialize file paths or a list of file names.
        
        self.transform = transform
        self.cv = cv
        self.diag_0 = []; self.diag_1 = []; self.diag_2 = []; self.diag_4 = []; self.diag_5 = []; self.diag_6 = []; self.diag_7 = []; self.diag_8 = [];
        #self.tumor_0 = []; self.tumor_1 = [];
        self.examples = [];
        self.train_tumor = [];
        self.train_nontumor = [];

        root_h5 = pathlib.Path(root) / 'H5_FINETUNED'
        files_h5 = sorted(list(pathlib.Path(root_h5).iterdir()))

        for i in tqdm(range(len(files_h5))):
            tumor = h5py.File(files_h5[i], 'r')['tumor'][()]
            diag = h5py.File(files_h5[i], 'r')['diagnosis'][()]
            inc = h5py.File(files_h5[i], 'r')['include'][()]
            seg_stat = h5py.File(files_h5[i], 'r')['seg_status'][()]

            if diag==0:
                self.diag_0.append(files_h5[i])
            elif diag==1:
                self.diag_1.append(files_h5[i])
            elif diag==2:
                self.diag_2.append(files_h5[i])
            elif diag==4:
                self.diag_4.append(files_h5[i])
            elif diag==5:
                self.diag_5.append(files_h5[i])
            elif diag==6:
                self.diag_6.append(files_h5[i])
            elif diag==7:
                self.diag_7.append(files_h5[i])
            elif diag==8:
                self.diag_8.append(files_h5[i])
                    
        print('DIAG_0: %d, DIAG_1: %d, DIAG_2: %d, DIAG_4: %d, DIAG_5: %d, DIAG_6: %d, DIAG_7: %d, DIAG_8: %d'%((len(self.diag_0), len(self.diag_1), len(self.diag_2), len(self.diag_4), len(self.diag_5), len(self.diag_6), len(self.diag_7), len(self.diag_8))))
        
        random.seed(42); random.shuffle(self.diag_0)
        random.seed(42); random.shuffle(self.diag_1)
        random.seed(42); random.shuffle(self.diag_2)
        random.seed(42); random.shuffle(self.diag_4)
        random.seed(42); random.shuffle(self.diag_5)
        random.seed(42); random.shuffle(self.diag_6)
        random.seed(42); random.shuffle(self.diag_7)
        random.seed(42); random.shuffle(self.diag_8)
        
        ## 5-fold CV indexing
        data_index_0 = np.arange(0, len(self.diag_0), 1).tolist() ; data_index_cv_0 = np.array_split(data_index_0, 5)
        data_index_1 = np.arange(0, len(self.diag_1), 1).tolist() ; data_index_cv_1 = np.array_split(data_index_1, 5)
        data_index_2 = np.arange(0, len(self.diag_2), 1).tolist() ; data_index_cv_2 = np.array_split(data_index_2, 5)
        data_index_4 = np.arange(0, len(self.diag_4), 1).tolist() ; data_index_cv_4 = np.array_split(data_index_4, 5)
        data_index_5 = np.arange(0, len(self.diag_5), 1).tolist() ; data_index_cv_5 = np.array_split(data_index_5, 5)
        data_index_6 = np.arange(0, len(self.diag_6), 1).tolist() ; data_index_cv_6 = np.array_split(data_index_6, 5)
        data_index_7 = np.arange(0, len(self.diag_7), 1).tolist() ; data_index_cv_7 = np.array_split(data_index_7, 5)
        data_index_8 = np.arange(0, len(self.diag_8), 1).tolist() ; data_index_cv_8 = np.array_split(data_index_8, 5)
        
        if self.cv=='1':
            idx_cv_0 = list(data_index_cv_0[0])+list(data_index_cv_0[1])+list(data_index_cv_0[2])+list(data_index_cv_0[3]) if challenge=='train' else list(data_index_cv_0[4])
            idx_cv_1 = list(data_index_cv_1[0])+list(data_index_cv_1[1])+list(data_index_cv_1[2])+list(data_index_cv_1[3]) if challenge=='train' else list(data_index_cv_1[4])
            idx_cv_2 = list(data_index_cv_2[0])+list(data_index_cv_2[1])+list(data_index_cv_2[2])+list(data_index_cv_2[3]) if challenge=='train' else list(data_index_cv_2[4])
            idx_cv_4 = list(data_index_cv_4[0])+list(data_index_cv_4[1])+list(data_index_cv_4[2])+list(data_index_cv_4[3]) if challenge=='train' else list(data_index_cv_4[4])
            idx_cv_5 = list(data_index_cv_5[0])+list(data_index_cv_5[1])+list(data_index_cv_5[2])+list(data_index_cv_5[3]) if challenge=='train' else list(data_index_cv_5[4])
            idx_cv_6 = list(data_index_cv_6[0])+list(data_index_cv_6[1])+list(data_index_cv_6[2])+list(data_index_cv_6[3]) if challenge=='train' else list(data_index_cv_6[4])
            idx_cv_7 = list(data_index_cv_7[0])+list(data_index_cv_7[1])+list(data_index_cv_7[2])+list(data_index_cv_7[3]) if challenge=='train' else list(data_index_cv_7[4])
            idx_cv_8 = list(data_index_cv_8[0])+list(data_index_cv_8[1])+list(data_index_cv_8[2])+list(data_index_cv_8[3]) if challenge=='train' else list(data_index_cv_8[4])
        elif self.cv=='2':
            idx_cv_0 = list(data_index_cv_0[0])+list(data_index_cv_0[1])+list(data_index_cv_0[2])+list(data_index_cv_0[4]) if challenge=='train' else list(data_index_cv_0[3])
            idx_cv_1 = list(data_index_cv_1[0])+list(data_index_cv_1[1])+list(data_index_cv_1[2])+list(data_index_cv_1[4]) if challenge=='train' else list(data_index_cv_1[3])
            idx_cv_2 = list(data_index_cv_2[0])+list(data_index_cv_2[1])+list(data_index_cv_2[2])+list(data_index_cv_2[4]) if challenge=='train' else list(data_index_cv_2[3])
            idx_cv_4 = list(data_index_cv_4[0])+list(data_index_cv_4[1])+list(data_index_cv_4[2])+list(data_index_cv_4[4]) if challenge=='train' else list(data_index_cv_4[3])
            idx_cv_5 = list(data_index_cv_5[0])+list(data_index_cv_5[1])+list(data_index_cv_5[2])+list(data_index_cv_5[4]) if challenge=='train' else list(data_index_cv_5[3])
            idx_cv_6 = list(data_index_cv_6[0])+list(data_index_cv_6[1])+list(data_index_cv_6[2])+list(data_index_cv_6[4]) if challenge=='train' else list(data_index_cv_6[3])
            idx_cv_7 = list(data_index_cv_7[0])+list(data_index_cv_7[1])+list(data_index_cv_7[2])+list(data_index_cv_7[4]) if challenge=='train' else list(data_index_cv_7[3])
            idx_cv_8 = list(data_index_cv_8[0])+list(data_index_cv_8[1])+list(data_index_cv_8[2])+list(data_index_cv_8[4]) if challenge=='train' else list(data_index_cv_8[3])
        elif self.cv=='3':
            idx_cv_0 = list(data_index_cv_0[0])+list(data_index_cv_0[1])+list(data_index_cv_0[3])+list(data_index_cv_0[4]) if challenge=='train' else list(data_index_cv_0[2])
            idx_cv_1 = list(data_index_cv_1[0])+list(data_index_cv_1[1])+list(data_index_cv_1[3])+list(data_index_cv_1[4]) if challenge=='train' else list(data_index_cv_1[2])
            idx_cv_2 = list(data_index_cv_2[0])+list(data_index_cv_2[1])+list(data_index_cv_2[3])+list(data_index_cv_2[4]) if challenge=='train' else list(data_index_cv_2[2])
            idx_cv_4 = list(data_index_cv_4[0])+list(data_index_cv_4[1])+list(data_index_cv_4[3])+list(data_index_cv_4[4]) if challenge=='train' else list(data_index_cv_4[2])
            idx_cv_5 = list(data_index_cv_5[0])+list(data_index_cv_5[1])+list(data_index_cv_5[3])+list(data_index_cv_5[4]) if challenge=='train' else list(data_index_cv_5[2])
            idx_cv_6 = list(data_index_cv_6[0])+list(data_index_cv_6[1])+list(data_index_cv_6[3])+list(data_index_cv_6[4]) if challenge=='train' else list(data_index_cv_6[2])
            idx_cv_7 = list(data_index_cv_7[0])+list(data_index_cv_7[1])+list(data_index_cv_7[3])+list(data_index_cv_7[4]) if challenge=='train' else list(data_index_cv_7[2])
            idx_cv_8 = list(data_index_cv_8[0])+list(data_index_cv_8[1])+list(data_index_cv_8[3])+list(data_index_cv_8[4]) if challenge=='train' else list(data_index_cv_8[2])
        elif self.cv=='4':
            idx_cv_0 = list(data_index_cv_0[0])+list(data_index_cv_0[2])+list(data_index_cv_0[3])+list(data_index_cv_0[4]) if challenge=='train' else list(data_index_cv_0[1])
            idx_cv_1 = list(data_index_cv_1[0])+list(data_index_cv_1[2])+list(data_index_cv_1[3])+list(data_index_cv_1[4]) if challenge=='train' else list(data_index_cv_1[1])
            idx_cv_2 = list(data_index_cv_2[0])+list(data_index_cv_2[2])+list(data_index_cv_2[3])+list(data_index_cv_2[4]) if challenge=='train' else list(data_index_cv_2[1])
            idx_cv_4 = list(data_index_cv_4[0])+list(data_index_cv_4[2])+list(data_index_cv_4[3])+list(data_index_cv_4[4]) if challenge=='train' else list(data_index_cv_4[1])
            idx_cv_5 = list(data_index_cv_5[0])+list(data_index_cv_5[2])+list(data_index_cv_5[3])+list(data_index_cv_5[4]) if challenge=='train' else list(data_index_cv_5[1])
            idx_cv_6 = list(data_index_cv_6[0])+list(data_index_cv_6[2])+list(data_index_cv_6[3])+list(data_index_cv_6[4]) if challenge=='train' else list(data_index_cv_6[1])
            idx_cv_7 = list(data_index_cv_7[0])+list(data_index_cv_7[2])+list(data_index_cv_7[3])+list(data_index_cv_7[4]) if challenge=='train' else list(data_index_cv_7[1])
            idx_cv_8 = list(data_index_cv_8[0])+list(data_index_cv_8[2])+list(data_index_cv_8[3])+list(data_index_cv_8[4]) if challenge=='train' else list(data_index_cv_8[1])
        else:
            idx_cv_0 = list(data_index_cv_0[1])+list(data_index_cv_0[2])+list(data_index_cv_0[3])+list(data_index_cv_0[4]) if challenge=='train' else list(data_index_cv_0[0])
            idx_cv_1 = list(data_index_cv_1[1])+list(data_index_cv_1[2])+list(data_index_cv_1[3])+list(data_index_cv_1[4]) if challenge=='train' else list(data_index_cv_1[0])
            idx_cv_2 = list(data_index_cv_2[1])+list(data_index_cv_2[2])+list(data_index_cv_2[3])+list(data_index_cv_2[4]) if challenge=='train' else list(data_index_cv_2[0])
            idx_cv_4 = list(data_index_cv_4[1])+list(data_index_cv_4[2])+list(data_index_cv_4[3])+list(data_index_cv_4[4]) if challenge=='train' else list(data_index_cv_4[0])
            idx_cv_5 = list(data_index_cv_5[1])+list(data_index_cv_5[2])+list(data_index_cv_5[3])+list(data_index_cv_5[4]) if challenge=='train' else list(data_index_cv_5[0])
            idx_cv_6 = list(data_index_cv_6[1])+list(data_index_cv_6[2])+list(data_index_cv_6[3])+list(data_index_cv_6[4]) if challenge=='train' else list(data_index_cv_6[0])
            idx_cv_7 = list(data_index_cv_7[1])+list(data_index_cv_7[2])+list(data_index_cv_7[3])+list(data_index_cv_7[4]) if challenge=='train' else list(data_index_cv_7[0])
            idx_cv_8 = list(data_index_cv_8[1])+list(data_index_cv_8[2])+list(data_index_cv_8[3])+list(data_index_cv_8[4]) if challenge=='train' else list(data_index_cv_8[0])
         
        for i, input_file in enumerate(self.diag_0):
            if i in idx_cv_0:
                self.examples += [str(input_file)]
                
        for i, input_file in enumerate(self.diag_1):
            if i in idx_cv_1:
                self.examples += [str(input_file)]
                
        for i, input_file in enumerate(self.diag_2):
            if i in idx_cv_2:
                self.examples += [str(input_file)]
        
        for i, input_file in enumerate(self.diag_4):
            if i in idx_cv_4:
                self.examples += [str(input_file)]
                
        for i, input_file in enumerate(self.diag_5):
            if i in idx_cv_5:
                self.examples += [str(input_file)]
                
        for i, input_file in enumerate(self.diag_6):
            if i in idx_cv_6:
                self.examples += [str(input_file)]
                
        for i, input_file in enumerate(self.diag_7):
            if i in idx_cv_7:
                self.examples += [str(input_file)]
                
        for i, input_file in enumerate(self.diag_8):
            if i in idx_cv_8:
                self.examples += [str(input_file)]
                
        print('LENGTH SELF.EXAMPLES:', len(self.examples)) #Should be 189+199=388
        
        random.seed(0)
        random.shuffle(self.examples)
        
    def __getitem__(self, i):
        
        input_h5 = self.examples[i]
        
        t1ce = np.transpose(h5py.File(input_h5, 'r')['t1ce'][()], (2,1,0)).astype(np.float32)
        t1 = np.transpose(h5py.File(input_h5, 'r')['t1'][()], (2,1,0)).astype(np.float32)
        flair = np.transpose(h5py.File(input_h5, 'r')['flair'][()], (2,1,0)).astype(np.float32)
        dwi = np.transpose(h5py.File(input_h5, 'r')['dwi'][()], (2,1,0)).astype(np.float32)
        adc = np.transpose(h5py.File(input_h5, 'r')['adc'][()], (2,1,0)).astype(np.float32)
        seg = np.transpose(h5py.File(input_h5, 'r')['seg'][()], (2,1,0)).astype(np.float32)
        t1ce_mask = np.transpose(h5py.File(input_h5, 'r')['t1ce_mask'][()], (2,1,0)).astype(np.bool)
        flair_mask = np.transpose(h5py.File(input_h5, 'r')['flair_mask'][()], (2,1,0)).astype(np.bool)
        tumor = h5py.File(input_h5, 'r')['tumor'][()].astype(np.float32)
        
        diagnosis = h5py.File(input_h5, 'r')['diagnosis'][()].astype(np.float32)
        
        if diagnosis==8:
            diagnosis = np.array([[3]]).astype(np.float32)
        
        return self.transform(t1, t1ce, flair, dwi, adc, seg, t1ce_mask, flair_mask, tumor, diagnosis)
        
    def __len__(self):
        
        # You should change 0 to the total size of your dataset(length of the list).
        return len(self.examples)
