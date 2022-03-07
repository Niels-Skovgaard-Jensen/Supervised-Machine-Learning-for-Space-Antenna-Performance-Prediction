from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch


class SingleCutDataset(Dataset):
    """Field Cut dataset"""

    def __init__(self,cut):
        """
        Args:
            cut (integer) : Integer name of file in directory
        """
        
        # Define data placement
        cut_dir = Path('AntennaDatasets/ReflectorAntennaSimpleDataset1/cut_files')
        param_dir = Path('AntennaDatasets/ReflectorAntennaSimpleDataset1/log_files')
        cut_file = cut_dir / (str(cut)+'.cut')
        param_file = param_dir / 'lookup.log'
        

        V_INI, V_INC, V_NUM, C, ICOMP, ICUT, NCOMP = np.genfromtxt(cut_file, max_rows=1, skip_header=1)
        self.thetas = torch.linspace(V_INI,V_INI+V_INC*(V_NUM-1),int(V_NUM))
        self.field_cut = np.genfromtxt(cut_file, skip_header=2,dtype = np.float32).T
        
        antenna_parameters = np.genfromtxt(param_file, skip_header=1,dtype = np.float32).T
        #antenna_parameters = antenna_parameters[:,cut]
        
        self.parameters = torch.Tensor([np.append(antenna_parameters[1:,cut],theta) for theta in self.thetas])

    def __len__(self):
        return self.field_cut.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        parameters = self.parameters[idx,:]
        field_val = self.field_cut[:,idx]
        
        if self.transform:
            sample = self.transform(parameters)
            
            
        return parameters, field_val
    
class ReflectorCutDataset(Dataset):
    """Field Cut dataset"""

    def __init__(self, cuts = 2499,flatten_output = False):
        """
        Args:
            cut (integer) : Integer name of file in directory
        """
        self.flatten_output = flatten_output
        # Define data placement
        cut_dir = Path('AntennaDatasets/ReflectorAntennaSimpleDataset1/cut_files')
        param_dir = Path('AntennaDatasets/ReflectorAntennaSimpleDataset1/log_files')
        param_file = param_dir / 'lookup.log'
        
        
        file_to_open = cut_dir / '0.cut'
        self.field_cut = np.genfromtxt(file_to_open, skip_header=2,dtype = np.float32).T.reshape(4,1001,1)


        
        for i in range(1,cuts):
            file_to_open = cut_dir / (str(i)+'.cut')
            openFileData = np.genfromtxt(file_to_open, skip_header=2,dtype = np.float32).T.reshape(4,1001,1)
            self.field_cut = np.append(self.field_cut,openFileData,axis = 2)
        
       
        
        self.antenna_parameters = np.genfromtxt(param_file, skip_header=1,dtype = np.float32).T
        #antenna_parameters = antenna_parameters[:,cut]
        

    def __len__(self):
        return self.field_cut.shape[2]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        parameters = torch.tensor(self.antenna_parameters[1:4,idx])
        if self.flatten_output:
            field_val = torch.tensor(self.field_cut[:,:,idx]).flatten()
        else:
            field_val = torch.tensor(self.field_cut[:,:,idx])

            
        return parameters, field_val