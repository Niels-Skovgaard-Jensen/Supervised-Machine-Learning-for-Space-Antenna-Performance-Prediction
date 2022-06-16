# Standard Packages
from re import A
from torch.utils.data import random_split, DataLoader
import torch
import wandb
from datetime import datetime
from matplotlib import pyplot as plt
import torch.nn as nn
# Custom Packages
from ssapp.data.AntennaDatasetLoaders import load_serialized_dataset,get_single_dataset_example
from ssapp.Utils import train_test_data_split
from ssapp.data.Metrics import relRMSE, relRMSE_pytorch
from ssapp.models.HelperFunctions import saveModel, saveConfig
from ssapp.models.NeuralNetworkModels.SimpleFeedForward import FCBenchmark,PDNN
import yaml

# General Settings
torch.manual_seed(42) # Manual seed for sanity
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG = {
    "learning_rate": 3e-4,
    "epochs": 300,
    "batch_size": 4,
    'num layers':4,
    'phi_k' : 400,
    's_c' : 1.2,
    'alpha': 0.01,
    'dataset': 'CircularHornDataset1',
    #'dataset': 'PatchAntennaDataset2',
    }

project = "PATCH_PDNN"

wandb.init(config = DEFAULT_CONFIG,project=project, entity="skoogy_dan")
CONFIG = wandb.config
run_name = str(wandb.run.name)

#dataset = load_serialized_dataset('CircularHornDataset1')

BATCH_SIZE = CONFIG['batch_size']
NUM_WORKERS = 4

# Load Serialized training and validation set
train_set = load_serialized_dataset(CONFIG['dataset']+'_Train')
val_set = load_serialized_dataset(CONFIG['dataset']+'_Val')

# Split validation set into validation and test set 
val_set,test_set = train_test_data_split(val_set,TRAIN_TEST_RATIO=0.8)

# Get single dataset example for NN dimensionality
params,fields = get_single_dataset_example(train_set)

#Define model
model = PDNN(input_size = params.shape[1],
            num_layers=CONFIG['num layers'],
            phi_k = CONFIG['phi_k'],
            s_c = CONFIG['s_c'],
            alpha = CONFIG['alpha'])



if torch.cuda.device_count() > 1:
  print("Multiple GPUs found, using:", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

# Create dataloaders for training, validation and test datasets
train_dataloader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True,num_workers = NUM_WORKERS,drop_last=True)
val_dataloader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS,drop_last=True)
test_dataloader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS,drop_last=True)


model = model.to(device)

criterion = relRMSE_pytorch() # Custom loss function for relative RMSE

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate']) 

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,150,200,250,300], verbose = True)

train_loss_array = []
test_loss_array = []
loss_array = []

train_size = len(train_set)
val_size = len(val_set)
test_size = len(test_set)

best_val_loss = float("inf")
for epoch in range(CONFIG["epochs"]):

    epoch_training_loss = 0
    epoch_training_targets = 0
    model.train()
    for train_params, train_fields in train_dataloader:
        batch_size = len(train_fields)
        ## Transfer Batch to Device
        train_params = train_params.float().to(device)
        train_fields = train_fields.float().to(device)
        
        prediction = model(train_params)
        
        #loss = criterion(input = prediction, target = train_fields)
        loss = criterion(prediction, train_fields)
        epoch_training_loss += loss*(batch_size/train_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Caluclate make validation evaluation
    epoch_val_loss = 0
    epoch_val_targets = 0
    with torch.no_grad():
        model.eval()
        for val_params, val_fields in val_dataloader:
            batch_size = len(val_fields) # To ensure end batches are also calculated correctly
            # Put thing to device
            val_params = val_params.float().to(device)
            val_fields = val_fields.float().to(device)
            
            
            prediction = model(val_params)
            #loss = criterion(input = prediction, target = test_fields)
            loss = criterion(prediction, val_fields)
            epoch_val_loss += loss*(batch_size/val_size)


    # Save model if validation loss is best
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        saveModel(model=model,name = run_name,subfolder=project)



    # Make test evaluation
    epoch_test_loss = 0
    epoch_test_targets = 0
    with torch.no_grad():
        model.eval()
        for test_params, test_fields in test_dataloader:
            batch_size = len(test_fields)
            # Put thing to device
            test_params = test_params.float().to(device)
            test_fields = test_fields.float().to(device)

            prediction = model(test_params)
            #loss = criterion(input = prediction, target = test_fields)
            loss = criterion(prediction, test_fields)
            epoch_test_loss += loss*(batch_size/val_size)

    





    scheduler.step()

    wandb.log({'Train_loss':epoch_training_loss,
                'Val_loss':epoch_val_loss,
                'Test_loss':epoch_test_loss,
                'best_val_loss':best_val_loss,
                'Learning Rate':scheduler.get_lr()})
    
    print("epoch : {}/{}, train_loss = {:.9e}, val_loss = {:.9e}".format(epoch + 1, CONFIG["epochs"], loss,epoch_val_loss))
    

print('Training time:', datetime.now()-startTime)

