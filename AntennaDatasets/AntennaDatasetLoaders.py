from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


class SingleCutDataset(Dataset):
    """Field Cut dataset"""

    def __init__(self,cut):
        """
        Args:
            cut (integer) : Integer name of file in directory
        """
        
        # Define data placement
        cut_dir = Path('../PatchAntennaDataset1/cut_files')
        param_dir = Path('../PatchAntennaDataset1/log_files')
        cut_file = cut_dir / (str(cut)+'.cut')
        param_file = param_dir / 'lookup.log'
        
        import numpy as np
        V_INI, V_INC, V_NUM, C, ICOMP, ICUT, NCOMP = np.genfromtxt(cut_file, max_rows=1, skip_header=1)
        self.thetas = torch.linspace(V_INI,V_INI+V_INC*(V_NUM-1),int(V_NUM))
        self.field_cut = np.genfromtxt(cut_file, skip_header=2,dtype = np.float32).T
        
        antenna_parameters = np.genfromtxt(param_file, skip_header=1,dtype = np.float32).T
        #antenna_parameters = antenna_parameters[:,cut]
        
        self.parameters = torch.Tensor([np.append(antenna_parameters[1:,cut],theta) for theta in thetas])

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