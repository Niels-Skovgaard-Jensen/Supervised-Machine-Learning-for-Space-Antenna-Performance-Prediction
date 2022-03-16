

from typing import Dict
from IPython import display 
from pathlib import Path
from matplotlib import pyplot as plt
import pylab as pl

from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from ssapp.models.NeuralNetworkModels.Autoencoders import PatchAntenna1ConvAutoEncoder
from ssapp.models.HelperFunctions import saveModel
from ssapp.data.AntennaDatasetLoaders import PatchAntennaDataset
from ssapp.Utils import train_test_data_split

import wandb
print('Running')
torch.manual_seed(42)

def train(model : torch.nn, CONFIG : Dict, train_dataloader: DataLoader,test_dataloader, optimizer,criterion):

    EPOCHS = CONFIG['epochs']
    BATCH_SIZE = CONFIG['batch_size']

    train_loss_array = []
    validation_loss_array = []
    for epoch in range(EPOCHS):
        loss = 0
        test_loss = 0
        for params, field in train_dataloader:
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            field = field.float().to(device)
            # compute reconstructions
            train_outputs = model(field)
                
            # compute training reconstruction loss
            train_loss = criterion(train_outputs, field)
                
            # compute accumulated gradients
            train_loss.backward()
                
            # perform parameter update based on current gradients
            optimizer.step()
                
            # add the mini-batch training loss to epoch loss
            loss += train_loss.detach().item()

        with torch.no_grad():
            for params,field in test_dataloader:
                field = field.float().to(device)
                
                val_outputs = model(field)
                
                test_loss += criterion(val_outputs, field)
                
        
        loss = loss/(len(train_dataloader))
        val_loss = test_loss/len(test_dataloader)
        wandb.log({'loss':loss,
                    'val_loss':val_loss})
        train_loss_array.append(loss)
        # display the epoch training loss
        print("epoch : {}/{}, train_loss = {:.9e}, val_loss = {:.9e}".format(epoch + 1, EPOCHS, loss,val_loss))

    return model

if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:',device)

    DEFAULT_CONFIG = {
    "learning_rate": 1e-3,
    "epochs": 500,
    "batch_size": 1,
    "latent_size": 2,
    "number_cuts" : 343,
    "random_seed" : 42,
    }

    wandb.init(config = DEFAULT_CONFIG,project="FarFieldAutoEncoder", entity="skoogy_dan")
    CONFIG = wandb.config
    run_name = wandb.run.name

    BATCH_SIZE = 1
    EPOCHS = CONFIG['epochs']
    CUTS = 343 #max 343
    LATENT_SIZE = CONFIG['latent_size']
    LEARNING_RATE = CONFIG['learning_rate']


    data = PatchAntennaDataset()
    train_data, test_data = train_test_data_split(data, TRAIN_TEST_RATIO = 0.7)

    train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)


    model = PatchAntenna1ConvAutoEncoder(Latent_size = LATENT_SIZE)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()


    model = train(model = model,
                CONFIG = CONFIG,
                train_dataloader= train_loader,
                test_dataloader=test_loader,
                optimizer=optimizer,
                criterion=criterion)
    

    saveModel(model, run_name)
