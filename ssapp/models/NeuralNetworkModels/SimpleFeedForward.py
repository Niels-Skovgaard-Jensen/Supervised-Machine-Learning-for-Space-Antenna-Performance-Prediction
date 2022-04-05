

from typing import OrderedDict
import torch.nn as nn


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
        node_list = 
        for i,num_nodes in enumerate(config['net_nodes']):
            self.regressor_architecture.append(nn.Linear(config['net_nodes'][i-1] ,config['net_nodes'][i]))


        self.regressor = nn.Sequential(OrderedDict(self.regressor_architecture))
        
    def forward(self, x):
        output = self.regressor(x)
        return output
    