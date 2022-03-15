

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

import wandb
print('Running')
torch.manual_seed(42)


if __name__ == "__main__":
    
    wandb.init(project="FarFieldAutoEncoder", entity="skoogy_dan")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:',device)

    run_name = wandb.run.name


    BATCH_SIZE = 1
    EPOCHS = 20
    CUTS = 343 #max 343
    LATENT_SIZE = 3
    LEARNING_RATE = 1e-3

    wandb.config = {
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "latent_size": LATENT_SIZE,
    "number_cuts" : CUTS,
    "random_seed" : 42,
    }

    data = PatchAntennaDataset()
    patch_dataloader = DataLoader(data,batch_size=BATCH_SIZE,shuffle=True)
    model = PatchAntenna1ConvAutoEncoder(Latent_size = LATENT_SIZE)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    # mean-squared error loss
    criterion = nn.MSELoss()
    loss_array = []
    for epoch in range(EPOCHS):
        loss = 0
        for params, field in patch_dataloader:
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            field = field.float().to(device)
            # compute reconstructions
            outputs = model(field)
                
            # compute training reconstruction loss
            train_loss = criterion(outputs, field)
                
            # compute accumulated gradients
            train_loss.backward()
                
            # perform parameter update based on current gradients
            optimizer.step()
                
            # add the mini-batch training loss to epoch loss
            loss += train_loss.detach().item()
        
        loss = loss/(len(patch_dataloader)*BATCH_SIZE)
        wandb.log({'loss':loss})
        loss_array.append(loss)
        # display the epoch training loss
        if epoch % (EPOCHS/10) == 0:

            print("epoch : {}/{}, loss = {:.9e}".format(epoch + 1, EPOCHS, loss))
        
    pth = Path().cwd()
    saveModel(model, run_name)
