from torch.utils.data import random_split, Dataset
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
import numpy as np
import torch
import random
import os
import pickle




def get_raw_dataset_path_for_single_parameter(dataset_name: str,subfolder):

    main_dir = Path().cwd().parents[1]
    data_dir = main_dir / 'data'
    subdict_dir = data_dir / 'raw'
    subdict_dir2 = subdict_dir / 'Data for plotting' / subfolder
    dataset_dir = subdict_dir2 / dataset_name

    cut_dir = dataset_dir / 'cut_files'
    log_dir = dataset_dir / "log_files" 

    return cut_dir, log_dir

class SingleParameterVarianceDataset(Dataset):


    def __init__(self,subfolder, dataset_name,num_params,cuts = 100):

        self.cuts = cuts
        self.name = dataset_name

        # Define data placement
        cut_dir, param_dir = get_raw_dataset_path_for_single_parameter(self.name,subfolder)
        param_file = param_dir / 'lookup.log'
        
        self.antenna_parameters = np.genfromtxt(param_file, skip_header=1,skip_footer=100-cuts,dtype = np.float32)
        self.antenna_parameters = self.antenna_parameters.reshape(cuts,num_params+1)[:,1:num_params+1]

        # Kinda hardcoded fix, might want to automate it a little more
        file_to_open = cut_dir / '0.cut'
        V_INI, V_INC, V_NUM, C, ICOMP, ICUT, NCOMP = np.genfromtxt(file_to_open, max_rows=1, skip_header=1)
        self.V_NUM = int(V_NUM)

        self.thetas = np.linspace(V_INI,V_INI+V_INC*(self.V_NUM-1),int(self.V_NUM))
        # Generate First Cut
        self.field_cut = np.genfromtxt(file_to_open, skip_header=2, max_rows= self.V_NUM).reshape(1,self.V_NUM,1,4)
        for i in range(1,3):
                self.field_cut=np.append(self.field_cut, np.genfromtxt(file_to_open, skip_header=2+i*(self.V_NUM+2), max_rows= self.V_NUM).reshape(1,self.V_NUM,1,4),axis=2)

        # Then append to that cut
        for i in range(1,cuts):
            file_to_open = cut_dir / (str(i)+'.cut')
            phi_cut = np.genfromtxt(file_to_open, skip_header=2, max_rows= self.V_NUM).reshape(1,self.V_NUM,1,4)
            for i in range(1,3):
                phi_cut=np.append(phi_cut, np.genfromtxt(file_to_open, skip_header=2+i*(self.V_NUM+2), max_rows= self.V_NUM).reshape(1,self.V_NUM,1,4),axis=2)

            self.field_cut = np.append(self.field_cut,phi_cut,axis = 0)
        
        ## Convert to tensors
        self.field_cut = torch.tensor(self.field_cut,dtype = torch.float)
        self.antenna_parameter = torch.tensor(self.antenna_parameters,dtype = torch.float)
    def __len__(self):
        return self.cuts
    


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        
        parameters = self.antenna_parameters[idx,:]



        field_val = self.field_cut[idx,:,:]

            
        return parameters, field_val