
from typing import Dict
from IPython import display 
from pathlib import Path
from matplotlib import pyplot as plt
import pylab as pl
import wandb

from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from ssapp.models.NeuralNetworkModels.Autoencoders import PatchAntenna1ConvAutoEncoder
from ssapp.models.NeuralNetworkModels.FFT_Neural_Nets import LatentSpaceNet
from ssapp.models.HelperFunctions import saveModel, loadModel
from ssapp.data.AntennaDatasetLoaders import PatchAntennaDataset
from ssapp.Utils import train_test_data_split


def latent_space_train(model : LatentSpaceNet,decoder:PatchAntenna1ConvAutoEncoder, CONFIG, train_dataloader: DataLoader,test_dataloader, optimizer,criterion):

    EPOCHS = CONFIG['epochs']
    BATCH_SIZE = CONFIG['batch_size']

    best_val_loss = float("inf")
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
            params = params.float().to(device)
            # compute reconstructions
            latent_space_guess = model(params)
            
            # Decode Latent Space Guess
            latent_space = decoder.encode(field)

            # compute training reconstruction loss
            train_loss = criterion(latent_space_guess, latent_space)
                
            # compute accumulated gradients
            train_loss.backward()
                
            # perform parameter update based on current gradients
            optimizer.step()
                
            # add the mini-batch training loss to epoch loss
            loss += train_loss.detach().item()

        with torch.no_grad():
            for params,field in test_dataloader:
                field = field.float().to(device)
                params = params.float().to(device)

                latent_space_guess = model(params)

                val_outputs = decoder.decode(latent_space_guess)
                
                test_loss += criterion(val_outputs, field)
                
        print('Latent Guess',latent_space_guess.shape())
        print('True Latent Space', val_outputs.shape())
        loss = loss/(len(train_dataloader))
        val_loss = test_loss/len(test_dataloader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_to_save = model

        wandb.log({'loss':loss,
                    'val_loss':val_loss,
                    'best_val_loss':best_val_loss})
        train_loss_array.append(loss)
        # display the epoch training loss
        print("epoch : {}/{}, train_loss = {:.9e}, val_loss = {:.9e}".format(epoch + 1, EPOCHS, loss,val_loss))

    return model, model_to_save

if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:',device)


    # Autoencoder Config
    AUTOENCODER_CONFIG = {
    "learning_rate": 4e-4,
    "epochs": 20000,
    "batch_size": 1,
    "latent_size": 2,
    "number_cuts" : 343,
    "random_seed" : 42,
    'coder_channel_1': 11,
    'coder_channel_2': 60,
    'cuts': 343}

    autoencoder = loadModel(PatchAntenna1ConvAutoEncoder(config = AUTOENCODER_CONFIG),'treasured-haze-99_best_val') # Good 2-dimensional latent space


    DEFAULT_CONFIG = {
    "learning_rate": 4e-4,
    "epochs": 100,
    "batch_size": 1,
    "latent_size": 2,
    "random_seed" : 42,
    'coder_channel_1': 8,
    'coder_channel_2': 16,
    'cuts': 343}

    wandb.init(config = DEFAULT_CONFIG,project="LatentSpaceRegression", entity="skoogy_dan")
    CONFIG = wandb.config
    run_name = wandb.run.name
    print('Applied Configuration:', CONFIG)

    data = PatchAntennaDataset(cuts = CONFIG['cuts'])
    train_data, test_data = train_test_data_split(data, TRAIN_TEST_RATIO = 0.7)

    train_loader = DataLoader(train_data,batch_size=CONFIG['batch_size'],shuffle=True)
    test_loader = DataLoader(test_data,batch_size=CONFIG['batch_size'],shuffle=True)


    decoder_model = PatchAntenna1ConvAutoEncoder(config = CONFIG)
    decoder_model.to(device)
    model = LatentSpaceNet()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()


    final_model,best_model = latent_space_train(model = model,
                decoder = decoder_model,
                CONFIG = CONFIG,
                train_dataloader= train_loader,
                test_dataloader=test_loader,
                optimizer=optimizer,
                criterion=criterion)
    

    #saveModel(final_model, run_name)
    #saveModel(best_model, run_name + '_best_val')
