


from IPython import display 
from pathlib import Path
from matplotlib import pyplot as plt
import pylab as pl

from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from ssapp.models.NeuralNetworkModels.Autoencoders import ConvAutoEncoderAndLatentRegressor
from ssapp.models.HelperFunctions import saveModel
from ssapp.data.AntennaDatasetLoaders import PatchAntennaDataset2, load_serialized_dataset
from ssapp.Utils import train_test_data_split
from ssapp.data.Metrics import relRMSE

import wandb
print('Running')
torch.manual_seed(42)

def train(model : torch.nn, CONFIG, train_dataloader: DataLoader,test_dataloader, optimizer,criterion):

    EPOCHS = CONFIG['epochs']
    BATCH_SIZE = CONFIG['batch_size']

    best_val_loss = float("inf")
    rec_train_loss_array = []
    pred_train_loss_array = []
    rec_val_loss_array = []
    pred_val_loss_array = []

    for epoch in range(EPOCHS):
        
        running_rec_loss = 0
        running_pred_loss = 0 
        running_num = 0
        for params, field in train_dataloader:

            optimizer.zero_grad()
            field = field.float().to(device)
            params = params.float().to(device)
            # compute reconstruction
            reconstruction = model.autoencode_train(field)
            # compute training reconstruction loss
            rec_loss = criterion(reconstruction, field)
            running_rec_loss += len(params)*rec_loss.detach().item()
            # backpropagate
            rec_loss.backward()
            # perform parameter update based on current gradients
            optimizer.step()

            # Train Parameter to Field model
            optimizer.zero_grad()
            
            # Field prediction from parameters
            field_pred = model(params)
            # Compute training field prediction loss
            pred_loss = criterion(field_pred, field)
            # Backpropagation
            pred_loss.backward()
                
            # add the mini-batch training loss to epoch loss
            running_pred_loss += len(params)*pred_loss.detach().item()
            running_num += len(params)


        rec_loss = running_rec_loss/running_num
        pred_loss = running_pred_loss/running_num

        running_rec_loss = 0
        running_pred_loss = 0 
        running_num = 0
        with torch.no_grad():
            for params,field in test_dataloader:

                optimizer.zero_grad()
                field = field.float().to(device)
                params = params.float().to(device)
                # compute reconstruction
                reconstruction = model.autoencode_train(field)
                # compute training reconstruction loss
                rec_loss = criterion(reconstruction, field)
                running_rec_loss += len(params)*rec_loss.detach().item()
                
                # Field prediction from parameters
                field_pred = model(params)
                # Compute training field prediction loss
                pred_loss = criterion(field_pred, field)
                    
                # add the mini-batch training loss to epoch loss
                running_pred_loss += len(params)*pred_loss.detach().item()
                running_num += len(params)
                
        
        rec_val_loss = running_rec_loss/running_num
        pred_val_loss = running_pred_loss/running_num

        if pred_val_loss < best_val_loss:
            best_val_loss = pred_val_loss
            model_best_val = model

        wandb.log({'pred_loss':pred_loss,
                    'pred_val_loss':pred_val_loss,
                    'rec_loss': rec_loss,
                    'red_val_loss': rec_val_loss,
                    'best_val_loss':best_val_loss})

        # display the epoch training loss
        print("epoch : {}/{}, train_loss = {:.9e}, val_loss = {:.9e}".format(epoch + 1, EPOCHS, rec_loss,rec_val_loss))
        print("epoch : {}/{}, train_loss = {:.9e}, val_loss = {:.9e}".format(epoch + 1, EPOCHS, pred_loss,pred_val_loss))

    return model, model_best_val

if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:',device)

    DEFAULT_CONFIG = {
    "learning_rate": 4e-4,
    "epochs": 200,
    "batch_size": 1,
    "latent_size": 5,
    "random_seed" : 42,
    'coder_channel_1': 16,
    'coder_channel_2': 32,
    'Parameter Number': 3}

    project = "FarFieldAutoEncoder"

    wandb.init(config = DEFAULT_CONFIG,project=project, entity="skoogy_dan")
    CONFIG = wandb.config
    run_name = wandb.run.name
    print('Applied Configuration:', CONFIG)

    data = load_serialized_dataset('PatchAntennaDataset2')
    train_data, test_data = train_test_data_split(data, TRAIN_TEST_RATIO = 0.7)

    train_loader = DataLoader(train_data,batch_size=CONFIG['batch_size'],shuffle=True)
    test_loader = DataLoader(test_data,batch_size=CONFIG['batch_size'],shuffle=True)

    model = ConvAutoEncoderAndLatentRegressor(config = CONFIG)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()


    final_model,best_model = train(model = model,
                CONFIG = CONFIG,
                train_dataloader= train_loader,
                test_dataloader=test_loader,
                optimizer=optimizer,
                criterion=criterion)
    

    saveModel(final_model, run_name, subfolder= project)
    saveModel(best_model, run_name + '_best_val')
