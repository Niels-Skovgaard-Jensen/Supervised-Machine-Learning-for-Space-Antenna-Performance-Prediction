
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
from ssapp.models.TrainingLoops import train



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
    "epochs": 2000,
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
    model = LatentSpaceNet()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()


    final_model,best_model = train(model = model,
                CONFIG = CONFIG,
                train_dataloader= train_loader,
                test_dataloader=test_loader,
                optimizer=optimizer,
                criterion=criterion)
    

    saveModel(final_model, run_name)
    saveModel(best_model, run_name + '_best_val')
