from torch.utils.data import random_split, Dataset
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
import numpy as np
import torch
import random
import os
import pickle






def set_global_random_seed(seed=42):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_raw_dataset_path(dataset_name: str):

    main_dir = Path().cwd().parents[1]
    data_dir = main_dir / 'data'
    subdict_dir = data_dir / 'raw'
    dataset_dir = subdict_dir / dataset_name

    cut_dir = dataset_dir / 'cut_files'
    log_dir = dataset_dir / "log_files" 

    return cut_dir, log_dir

def get_processed_dataset_path(dataset_name: str,ekstra_back_steps = 0):

    main_dir = Path().cwd().parents[1+ekstra_back_steps]
    data_dir = main_dir / 'data'
    subdict_dir = data_dir / 'processed'
    file_dir = subdict_dir / (dataset_name+'.pickle')

    return file_dir

def gen_coords_from_header(V_INI, V_INC,V_NUM, ):

    thetas = np.linspace(V_INI,V_INI+V_INC*(V_NUM-1),int(V_NUM))
    phis = None
    return thetas, phis


def serialize_dataset(dataset: Dataset,name = None):    

    save_dir = get_processed_dataset_path(dataset.name)

    with open(save_dir,'wb') as f:
        pickle.dump(dataset,f)
        f.close()

def dataset_is_serialized(dataset_name: str):
    
    serial_dir = get_processed_dataset_path(dataset_name) 

    if serial_dir.is_file():
        return True
    return False

def load_serialized_dataset(dataset_name: str,extra_back_steps = 0):

    load_dir = get_processed_dataset_path(dataset_name,extra_back_steps)
    with open(load_dir,'rb') as f:
        dataset = pickle.load(f)
        f.close()
    return dataset

def serialise_all_datasets(split = None):
    assert split in ['holdout',None,'k-fold']
    
    dataset_constructors = [PatchAntennaDataset,
                PatchAntennaDataset2,
                ReflectorCutDataset,
                ReflectorCutDataset2,
                CircularHornDataset1,
                MLADataset1,
                ]

    for dataset_constructor in dataset_constructors:

        #Instantiate dataset
        dataset = dataset_constructor()
        print('Serializing',dataset.name)
        # Make split if nessesary
        if type(split) == type(None):
            serialize_dataset(dataset)
        elif split == 'holdout':
            
            print('Dataset loaded:',dataset)
            train_dataset, val_dataset = train_test_data_split(dataset)
            train_dataset.name = dataset.name+'_Train'
            val_dataset.name = dataset.name+'_Val'
        elif split == 'k-fold':
            pass
        elif split == 'all':
            serialise_all_datasets()

def train_test_data_split(dataset, TRAIN_TEST_RATIO = 0.7, set_random_seed = True):

    train_len = int(len(dataset)*TRAIN_TEST_RATIO)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])    
    return train_set, test_set

def train_test_dataloader_split(dataset, batch_size, TRAIN_TEST_RATIO = 0.7, set_random_seed = True):


    train_set, test_set = train_test_data_split(dataset, TRAIN_TEST_RATIO, set_random_seed)
    if batch_size == None:
        train_dataloader = DataLoader(train_set, batch_size = len(train_set), shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size = len(test_set), shuffle=True)
    else:
        train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def get_single_dataset_example(dataset):

    train_dataloader, test_dataloader = train_test_dataloader_split(dataset, batch_size=1)
    return next(iter(test_dataloader))
    
    


class SingleCutDataset(Dataset):
    """Field Cut dataset"""

    def __init__(self,cut: int):
        """
        Args:
            cut (integer) : Integer name of file in directory
        """
        
        # Define data placement
        cut_dir, param_dir = get_raw_dataset_path('ReflectorAntennaSimpleDataset1')
        cut_file = cut_dir / (str(cut)+'.cut')
        param_file = param_dir / 'lookup.log'
        

        V_INI, V_INC, V_NUM, C, ICOMP, ICUT, NCOMP = np.genfromtxt(cut_file, max_rows=1, skip_header=1)
        self.thetas = torch.linspace(V_INI,V_INI+V_INC*(V_NUM-1),int(V_NUM))
        self.field_cut = np.genfromtxt(cut_file, skip_header=2,dtype = np.float32).T
        
        antenna_parameters = np.genfromtxt(param_file, skip_header=1,dtype = np.float32).T
        
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
        self.name = 'ReflectorCutDataset'
        # Define data placement
        cut_dir, param_dir = get_raw_dataset_path('ReflectorAntennaSimpleDataset1')
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
    
class ReflectorCutDataset2(Dataset):
    """
    To use for loading and parsing the ReflectorAntennaSimpleDataset2"""

    def __init__(self, cuts = 360,
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
        self.name = 'ReflectorCutDataset2'

        # Define data placement
        cut_dir, param_dir = get_raw_dataset_path('ReflectorAntennaSimpleDataset2')
        param_file = param_dir / 'lookup.log'
        
        self.antenna_parameters = np.genfromtxt(param_file, skip_header=1,skip_footer=360-cuts,dtype = np.float32)
        self.antenna_parameters = self.antenna_parameters.reshape(cuts,4)[:,1:4]

        ## Kinda hardcoded fix, might want to automate it a little more
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
        ## Apply to device?

    def __len__(self):
        return self.cuts
    
    def to(self,device):
        self.field_


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        parameters = self.antenna_parameters[idx,:]
        
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
                 co_polar_only = False,
                 mag_phase_transform = False):
        """
        Args:
            cut (integer) : Integer name of file in directory
            flatten_output (Boolean) : Flattens output into [cuts, 1001]
            magnitude_only : Only return magnitude of the complex dataset
            standardized_parameters : Standarizes the dataset for every parameter
        """
        super().__init__(cuts = cuts, flatten_output = flatten_output)
        self.co_polar_only = co_polar_only
        self.mag_phase_transform = mag_phase_transform

        self.co_polar_complex = torch.view_as_complex(self.field_cut[:,:,0:2]).reshape(-1,1001,1)
        self.x_polar_complex = torch.view_as_complex(self.field_cut[:,:,2:4]).reshape(-1,1001,1)
        self.field_cut = torch.cat((self.co_polar_complex,self.x_polar_complex),2)
        
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        parameters = self.antenna_parameters[idx,:]

        
        if self.mag_phase_transform:
            magnitude = torch.abs(self.field_cut[idx,:,:])
            phase = torch.angle(self.field_cut[idx,:,:])
            return parameters, magnitude, phase
        else:
            field_val = self.field_cut[idx,:,:]
            return parameters, field_val
    


class PatchAntennaDataset(Dataset):
    """
    To use for loading and parsing the ReflectorAntennaSimpleDataset1"""

    def __init__(self, cuts = 343,
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
        self.name = 'PatchAntennaDataset1'

        # Define data placement
        cut_dir, param_dir = get_raw_dataset_path(self.name)
        param_file = param_dir / 'lookup.log'
        
        self.antenna_parameters = np.genfromtxt(param_file, skip_header=1,skip_footer=343-cuts,dtype = np.float32)
        self.antenna_parameters = self.antenna_parameters.reshape(cuts,4)[:,1:4]

        ## Kinda hardcoded fix, might want to automate it a little more
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
        ## Apply to device?

    def __len__(self):
        return self.cuts
    


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        parameters = self.antenna_parameters[idx,:]
        
        
        if self.flatten_output:
            field_val = self.field_cut[idx,:,:].flatten()
        else:
            field_val = self.field_cut[idx,:,:]

            
        return parameters, field_val

class PatchAntennaDataset2(Dataset):
    """
    To use for loading and parsing the ReflectorAntennaSimpleDataset1"""

    def __init__(self, cuts = 3374,
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
        self.name = 'PatchAntennaDataset2'

        # Define data placement
        cut_dir, param_dir = get_raw_dataset_path(self.name)
        param_file = param_dir / 'lookup.log'
        
        self.antenna_parameters = np.genfromtxt(param_file, skip_header=1,skip_footer=3375-cuts,dtype = np.float32)
        print(self.antenna_parameters.shape)
        self.antenna_parameters = self.antenna_parameters.reshape(cuts,4)[:,1:4]

        ## Kinda hardcoded fix, might want to automate it a little more
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
        ## Apply to device?

    def __len__(self):
        return self.cuts
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        parameters = self.antenna_parameters[idx,:]
        
        
        if self.flatten_output:
            field_val = self.field_cut[idx,:,:].flatten()
        else:
            field_val = self.field_cut[idx,:,:]

            
        return parameters, field_val


class PatchAntennaDatasetComplex(PatchAntennaDataset):



    def __init__(self, cuts = 343,
                 flatten_output = False,
                 standardized_parameters = False,
                 mag_phase_transform = False):
    

        super().__init__(cuts = cuts, flatten_output = flatten_output)
        
        self.mag_phase_transform = mag_phase_transform

        self.co_polar_complex = torch.view_as_complex(self.field_cut[:,:,:,0:2]).reshape(-1,self.V_NUM,3,1)
        self.x_polar_complex = torch.view_as_complex(self.field_cut[:,:,:,2:4]).reshape(-1,self.V_NUM,3,1)
        self.field_cut = torch.cat((self.co_polar_complex,self.x_polar_complex),3)





    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        parameters = self.antenna_parameters[idx,:]

        if self.mag_phase_transform:
            magnitude = torch.abs(self.field_cut[idx,::])
            phase = torch.angle(self.field_cut[idx,::])
            return parameters, magnitude, phase
        else:
            field_val = self.field_cut[idx,::]
            return parameters, field_val

class CircularHornDataset1(Dataset):

    def __init__(self,cuts = 4000):
        self.cuts = cuts
        self.name = 'CircularHornDataset1'

        # Define data placement
        cut_dir, param_dir = get_raw_dataset_path(self.name)
        param_file = param_dir / 'lookup.log'
        
        self.antenna_parameters = np.genfromtxt(param_file, skip_header=1,skip_footer=4000-cuts,dtype = np.float32)
        self.antenna_parameters = self.antenna_parameters.reshape(cuts,3)[:,1:3]

        ## Kinda hardcoded fix, might want to automate it a little more
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


class MLADataset1(Dataset):

    def __init__(self,cuts = 10000):
        self.cuts = cuts
        self.name = 'MLADataset1'

        # Define data placement
        cut_dir, param_dir = get_raw_dataset_path(self.name)
        param_file = param_dir / 'lookup.log'
        
        self.antenna_parameters = np.genfromtxt(param_file, skip_header=1,skip_footer=10000-cuts,dtype = np.float32)
        self.antenna_parameters = self.antenna_parameters.reshape(cuts,11)[:,1:11]

        ## Kinda hardcoded fix, might want to automate it a little more
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

        assert type(parameters) == type(torch.tensor([]))

        field_val = self.field_cut[idx,:,:]

            
        return parameters, field_val



