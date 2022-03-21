# -*- coding: utf-8 -*-
"""
FFTNet
"""
import torch.nn as nn


class FFTNet(nn.Module):
    def __init__(self,NN):
        super(FFTNet, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(3, NN),
                                       nn.Tanh(),
                                       nn.Linear(NN, NN),
                                       nn.Tanh(),
                                       nn.Linear(NN, 4))
        
    def forward(self, x):
        output = self.regressor(x)
        return output

    
class LatentSpaceNet(nn.module):
    def __init__(self,NN = 10):
        super(FFTNet, self).__init__()

        self.regressor = nn.Sequential(nn.Linear(3, NN),
                                       nn.Tanh(),
                                       nn.Linear(NN, NN*2),
                                       nn.Tanh(),
                                       nn.Linear(NN*2, NN),
                                       nn.Tanh(),
                                       nn.Linear(NN, 2))



    def forward(self,x):
        output = self.regressor(x)
        return output
