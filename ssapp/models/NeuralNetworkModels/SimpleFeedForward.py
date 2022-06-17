

from re import A
from typing import OrderedDict
from torch import batch_norm
import torch
import torch.nn as nn
import numpy as np



class FCBenchmark(nn.Module):
    def __init__(self,input_size = 3):
        super(FCBenchmark, self).__init__()
        NN = 1000
        self.regressor = nn.Sequential(nn.Linear(input_size, NN*4),
                                       nn.LeakyReLU(),
                                       nn.Linear(NN*4, NN*2),
                                       nn.LeakyReLU(),
                                       nn.Linear(NN*2, int(NN*1.5)),
                                       nn.LeakyReLU(),
                                       nn.Linear(int(NN*1.5), 361*3*4))
        
    def forward(self, x):
        output = self.regressor(x)
        output = output.reshape(-1,361,3,4)
        return output

class DirectFeedForwardNet(nn.Module):
    def __init__(self,in_features,out_features,NN):
        super(DirectFeedForwardNet, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(in_features, NN),
                                       nn.ReLU(),
                                       nn.Linear(NN, NN),
                                       nn.ReLU(),
                                       nn.Linear(NN, out_features))
        
    def forward(self, x):
        output = self.regressor(x)
        return output

class DirectFeedForwardNet2(nn.Module):
    def __init__(self,in_features,out_features,NN = [10,20,10]):
        super(DirectFeedForwardNet2, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(in_features, NN[0]),
                                       nn.ReLU(),
                                       nn.Linear(NN[0], NN[1]),
                                       nn.ReLU(),
                                       nn.Linear(NN[1], NN[2]),
                                       nn.ReLU(),
                                       nn.Linear(NN[2], out_features))
        
    def forward(self, x):
        output = self.regressor(x)
        return output
    

class ConfigurableFFN(nn.Module):
    def __init__(self,in_features = 3,config = {}):
        super(ConfigurableFFN, self).__init__()

        default_config ={
                    'activation': nn.ReLU,
                    'net_nodes' : [10,10,10],
                    'latent_space_size': 2,

                }
        
        for config_param in default_config:
            if not bool(config[config_param]): # Does config contain the parameter?
                config[config_param] = default_config[config_param]


        self.regressor_architecture = []

        for i,num_nodes in enumerate(config['net_nodes']):
            self.regressor_architecture.append(nn.Linear(config['net_nodes'][i-1] ,config['net_nodes'][i]))


        self.regressor = nn.Sequential(OrderedDict(self.regressor_architecture))
        
    def forward(self, x):
        output = self.regressor(x)
        return output
    


class PDNN(nn.Module):
    """
    phi_k - number of neurons in last layer
    s_c - Triangle Scaling factor, higher means more pyramidal, 1 is flat
    alpha - leakyReLU leak factor

    This is a pyramidal neural network inspired by the paper
    [1] Accurate Modeling of Antenna Structures by Means of Domain Confinement and Pyramidal Deep Neural Networks
    by
    Slawomir Koziel, Nurullah Çalik, Peyman Mahouti, and Mehmet A. Belen
    """
    def __init__(self,input_size = 3,
                num_layers= 3,
                phi_k = 64,
                s_c = 1.2,
                alpha = 0.01,
                output_size= 361*3*4):
        super(PDNN, self).__init__()

        # Calculate size of layers from eq.3 [1]
        layer_count = lambda layer : int(np.ceil(phi_k*(s_c**(num_layers-layer))))
        modules = []
        layer_sizes = [layer_count(i) for i in range(1,num_layers+1)]

        print(layer_sizes)

        modules.append(nn.Linear(input_size,layer_sizes[0]))
        modules.append(nn.LeakyReLU(negative_slope=alpha))

        for i in range(1,len(layer_sizes)):


            modules.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            modules.append(nn.LeakyReLU(negative_slope=alpha))

        modules.append(nn.Linear(layer_sizes[-1],output_size))
        modules.append(nn.LeakyReLU(negative_slope=alpha))

        self.regressor = nn.Sequential(*modules)
        
    def forward(self, x):
        batch_size = len(x)
        output = self.regressor(x)
        output = output.reshape(batch_size,361,3,4)
        return output

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

        
class FCResNetBlock(nn.Module):
    """
    Implementation of fully connected Residual Network block
    """
    pass
    


class ResNetPDRN(nn.Module):
    """
    phi_k - number of neurons in last layer
    s_c - Triangle Scaling factor, higher means more pyramidal, 1 is flat
    alpha - leakyReLU leak factor


    This is a pyramidal neural network inspired by the paper
    [1] Accurate Modeling of Antenna Structures by Means of Domain Confinement and Pyramidal Deep Neural Networks
    by
    Slawomir Koziel, Nurullah Çalik, Peyman Mahouti, and Mehmet A. Belen
    """
    
    def __init__(self,input_size = 3,
                num_layers= 3,
                phi_k = 64,
                s_c = 1.2,
                alpha = 0.01,
                output_size= 361*3*4):
        super(ResNetPDRN, self).__init__()

        # Calculate size of layers from eq.3 [1]
        layer_count = lambda layer : int(np.round(phi_k*(s_c**(num_layers-layer))))
        modules = []
        layer_sizes = [layer_count(i) for i in range(1,num_layers+1)]

        print(layer_sizes)

        modules.append(nn.Linear(input_size,layer_sizes[0]))
        modules.append(nn.LeakyReLU(negative_slope=alpha))

        for i in range(1,len(layer_sizes)):


            modules.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            modules.append(nn.LeakyReLU(negative_slope=alpha))

        modules.append(nn.Linear(layer_sizes[-1],output_size))
        modules.append(nn.LeakyReLU(negative_slope=alpha))

        self.regressor = nn.Sequential(*modules)
        
    def forward(self, x):
        batch_size = len(x)
        output = self.regressor(x)
        output = output.reshape(batch_size,361,3,4)
        return output