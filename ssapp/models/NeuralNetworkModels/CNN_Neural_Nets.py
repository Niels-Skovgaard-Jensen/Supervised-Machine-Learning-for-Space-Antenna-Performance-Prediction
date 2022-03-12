

import torch.nn as nn


class CNNNet1(nn.Module):
    def __init__(self,NN):
        super(FFTNet, self).__init__()
        self.regressor = nn.Sequential(nn.Conv1d(3, NN))
        
    def forward(self, x):
        output = self.regressor(x)
        return output

    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(np.mean(copolar))
        