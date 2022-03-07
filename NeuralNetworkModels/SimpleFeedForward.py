

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

    

        