# -*- coding: utf-8 -*-
"""
FFTNet
"""
import torch.nn as nn




    
class LatentSpaceNet(nn.Module):
    def __init__(self,NN = 10):
        super(LatentSpaceNet, self).__init__()

        self.regressor = nn.Sequential(nn.Linear(3, NN),
                                       nn.ReLU(),
                                       nn.Linear(NN, NN*2),
                                       nn.ReLU(),
                                       nn.Linear(NN*2, NN),
                                       nn.ReLU(),
                                       nn.Linear(NN, 2))



    def forward(self,x):
        output = self.regressor(x)
        return output
