
from ssapp.models.NeuralNetworkModels.variational_autoencoder import VAE
from ssapp.models.NeuralNetworkModels.Autoencoders import AutoencoderFullyConnected,AutoencoderFCResNet
from ssapp.data.AntennaDatasetLoaders import load_serialized_dataset
from ssapp.Utils import train_test_data_split
from torch.utils.data.dataloader import DataLoader
from ssapp.data.Metrics import relRMSE_pytorch
from ssapp.models.HelperFunctions import saveModel
import torch.optim as optim
import torch.nn as nn
import torch
import wandb

from matplotlib import pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device',device)


#dataset = load_serialized_dataset('PatchAntennaDataset2',extra_back_steps=0)
train_dataset = load_serialized_dataset('CircularHornDataset1_Train',extra_back_steps=0)
val_dataset = load_serialized_dataset('CircularHornDataset1_Val',extra_back_steps=0)

train_len = len(train_dataset)
val_len = len(val_dataset)
DEFAULT_CONFIG = {'latent_size': 2,
            'model': 'AEFC',
            'batch_size' : 16}

# WeightAndBiases

wandb.init(config = DEFAULT_CONFIG,project="FarFieldAutoencoder2", entity="skoogy_dan")
CONFIG = wandb.config
run_name = wandb.run.name

train_dataloader = DataLoader(train_dataset,batch_size=CONFIG['batch_size'])
val_dataloader = DataLoader(val_dataset,batch_size=CONFIG['batch_size'])




#model = VAE(config)
if CONFIG['model'] == 'AEFC':
    model = AutoencoderFullyConnected(latent_size = CONFIG['latent_size'])
elif CONFIG['model'] == 'AEFCResNet':
    model = AutoencoderFCResNet(latent_size = CONFIG['latent_size'])

model.to(device)
model.double()

optimizer = optim.Adam(model.parameters(), lr=3e-4)


scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,200,300,400,500,600,700,800], verbose = True,gamma=0.5)

GAMMA = torch.log(0.5)/torch.log()
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=- 1, verbose=False)



criterion = relRMSE_pytorch

BETA_REC = 1   # Loss multiplication factor for reconstruction loss
BETA_SMOOTH = 0



best_val_loss = float("inf")
EPOCHS=500
for epoch in range(EPOCHS):
    train_epoch_loss, train_epoch_rec_loss = (0,0)

    train_epoch_smooth_loss = 0
    for train_params, train_fields in train_dataloader:
        batch_size = len(train_fields)

        train_fields= train_fields.to(device)

        optimizer.zero_grad()
        reconstruction = model(train_fields)
        rec_loss = criterion(reconstruction, train_fields)
        
        
        smooth_loss = torch.abs(torch.diff(reconstruction, dim = 1)).sum(dim=1).sum()

        rec_loss = rec_loss

        
        loss = rec_loss*BETA_REC+smooth_loss*BETA_SMOOTH
        loss.backward()
        
        train_epoch_smooth_loss += (smooth_loss/train_len)*batch_size
        train_epoch_rec_loss += (rec_loss/train_len)*batch_size
        train_epoch_loss += (loss/train_len)*batch_size

        optimizer.step()
    
    

    val_epoch_loss, val_epoch_rec_loss = (0,0)
    with torch.no_grad():
        
        for test_params, test_fields in val_dataloader:
            batch_size = len(test_fields)

            test_fields = test_fields.to(device)
            
            reconstruction = model(test_fields)
            rec_loss = criterion(reconstruction, test_fields)

            rec_loss = rec_loss
            loss = rec_loss*BETA_REC

            val_epoch_rec_loss += (rec_loss/val_len)*batch_size
            val_epoch_loss += (loss/val_len)*batch_size

    
    
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        saveModel(model=model,name = run_name+'best_val',subfolder='FarFieldAutoEncoder')

    wandb.log({'Training Rec. Loss':train_epoch_rec_loss,
                'Validation Rec. Loss':val_epoch_rec_loss,
                'Best Validation Loss':best_val_loss})
        
    scheduler.step()
    print("epoch : {}/{}, train_total_loss = {:.5f}%, train_rec_loss = {:.5f}%".format(epoch + 1, EPOCHS, train_epoch_loss*100,train_epoch_rec_loss*100))
    print("epoch : {}/{}, val_total_loss = {:.5f}%, val_rec_loss = {:.5f}%".format(epoch + 1, EPOCHS, val_epoch_loss*100,val_epoch_rec_loss*100))

saveModel(model=model,name = run_name+'_final',subfolder='FarFieldAutoEncoder')