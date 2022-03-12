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
        main_dir = Path().cwd().parents[1]
        data_dir = main_dir / 'data'
        raw_dir = data_dir / 'raw'
        
        param_dir = raw_dir / 'PatchAntennaDataset1' /  'log_files'
        cut_dir = raw_dir / 'PatchAntennaDataset1' /  'cut_files'

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
    """Reflector Dataset
    To use for loading and parsing the ReflectorAntennaSimpleDataset1"""

    def __init__(self, cuts = 2499,
                 flatten_output = False,
                 standardized_parameters = False):
        """
        Args:
            cut (integer) : Integer name of file in directory
            flatten_output (Boolean) : Flattens output into [cuts, 4004]
            standardized_parameters : Standarizes the dataset for every parameter
            
        """
        self.cuts = cuts
        self.flatten_output = flatten_output
        # Define data placement
        cut_dir = Path('AntennaDatasets/ReflectorAntennaSimpleDataset1/cut_files')
        param_dir = Path('AntennaDatasets/ReflectorAntennaSimpleDataset1/log_files')
        param_file = param_dir / 'lookup.log'
        
        self.antenna_parameters = np.genfromtxt(param_file, skip_header=1,skip_footer=2500-cuts,dtype = np.float32)
        self.antenna_parameters = self.antenna_parameters.reshape(cuts,4)[:,1:4]
        
        file_to_open = cut_dir / '0.cut'
        self.field_cut = np.genfromtxt(file_to_open, skip_header=2,dtype = np.float32).reshape(1,1001,4)
        for i in range(1,cuts):
            file_to_open = cut_dir / (str(i)+'.cut')
            openFileData = np.genfromtxt(file_to_open, skip_header=2,dtype = np.float32).reshape(1,1001,4)
            self.field_cut = np.append(self.field_cut,openFileData,axis = 0)
        
        ## Convert to tensors
        self.field_cut = torch.tensor(self.field_cut)
        self.antenna_parameter = torch.tensor(self.antenna_parameters)
        ## Apply to device?

    def __len__(self):
        return self.cuts
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        parameters = torch.tensor(self.antenna_parameters[idx,:])
        
        
        if self.flatten_output:
            field_val = self.field_cut[idx,:,:].flatten()
        else:
            field_val = self.field_cut[idx,:,:]

            
        return parameters, field_val
    
class ReflectorCutDatasetComplex(ReflectorCutDataset):
    """Reflector Dataset Complex representation
    To use for loading and parsing the ReflectorAntennaSimpleDataset1"""

    def __init__(self, cuts = 2499,
                 flatten_output = False,
                 standardized_parameters = False,
                 co_polar_only = False):
        """
        Args:
            cut (integer) : Integer name of file in directory
            flatten_output (Boolean) : Flattens output into [cuts, 2002]
            magnitude_only : Only return magnitude of the complex dataset
            standardized_parameters : Standarizes the dataset for every parameter
            
        """
        super().__init__(cuts = cuts, flatten_output = flatten_output)
        self.co_polar_only = co_polar_only
        self.co_polar_complex = torch.view_as_complex(self.field_cut[:,:,0:2]).reshape(-1,1001,1)
        self.x_polar_complex = torch.view_as_complex(self.field_cut[:,:,2:4]).reshape(-1,1001,1)
        self.field_cut = torch.cat((co_polar_complex,x_polar_complex),2)
        
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        parameters = self.antenna_parameters[idx,:]
        field_val = self.field_cut[idx,:,:]
        
        return parameters, mag_val
    

    
class ReflectorCutDatasetMagPhase(ReflectorCutDataset):
    """Reflector Dataset
    To use for loading and parsing the ReflectorAntennaSimpleDataset1"""

    def __init__(self, cuts = 2499,
                 flatten_output = False,
                 standardized_parameters = False):
        """
        Args:
            cut (integer) : Integer name of file in directory
            flatten_output (Boolean) : Flattens output into [cuts, 2002]
            magnitude_only : Only return magnitude of the complex dataset
            standardized_parameters : Standarizes the dataset for every parameter
            
        """
        super().__init__(cuts = cuts, flatten_output = flatten_output)
        
        real_co_polar = torch.tensor(self.field_cut[:,0,:])
        imag_co_polar = torch.tensor(self.field_cut[:,1,:])

        
        real_x_polar = torch.tensor(self.field_cut[:,2,:])
        imag_x_polar = torch.tensor(self.field_cut[:,3,:])
        
        
        
        co_polar_mag = torch.sqrt(self.field_cut[:,0,:]**2 + self.field_cut[:,1,:]**2).reshape(-1,1,1001)
        x_polar_mag = torch.sqrt(self.field_cut[:,2,:]**2 + self.field_cut[:,3,:]**2).reshape(-1,1,1001)
        
        #co_polar_phase = torc.angle
        #x_polar_phase = 
        
        self.mag_val = np.append(co_polar_mag,x_polar_mag,axis=2)
        self.phase_val = np.append(co_polar_phase, x_polar_phase)

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

            
        return parameters, mag_val, phase_val 
    
