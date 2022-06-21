# Standard Packages
from pickle import TRUE
from re import A
from torch.utils.data import random_split, DataLoader
import torch
import wandb
from datetime import datetime
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import time
# Custom Packages
from ssapp.data.AntennaDatasetLoaders import load_serialized_dataset,get_single_dataset_example
from ssapp.Utils import train_test_data_split
from ssapp.data.Metrics import relRMSE, relRMSE_pytorch
from ssapp.models.HelperFunctions import saveModel, saveConfig
from ssapp.models.NeuralNetworkModels.SimpleFeedForward import FCBenchmark,PDNN
import yaml
import time



# General Settings
torch.manual_seed(42) # Manual seed for sanity
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG = {
    "learning_rate": 3e-4,
    "epochs": 500,
    "batch_size": 4,
    'num_layers':6,
    'phi_k' : 1500,
    's_c' : 1.2,
    'alpha': 0.00,
    'dataset': 'CircularHornDataset1',
    #'dataset': 'PatchAntennaDataset2',
    #'dataset': 'ReflectorCutDataset2',
    #'dataset': 'MLADataset1'
    }

# Map used dataset to W&B project names
if DEFAULT_CONFIG['dataset'] == 'CircularHornDataset1':
    project = "CHA_PDNN"
elif DEFAULT_CONFIG['dataset'] == 'PatchAntennaDataset2':
    project = 'PATCH_PDNN'
elif DEFAULT_CONFIG['dataset'] == 'ReflectorCutDataset2':
    project = 'RFLCT_PDNN'
elif DEFAULT_CONFIG['dataset'] == 'MLADataset1':
    project = 'MLA_PDNN'
else:
    project = 'Def_PDNN'

wandb.init(config = DEFAULT_CONFIG,project=project, entity="skoogy_dan")
CONFIG = wandb.config
run_name = str(wandb.run.name)

BATCH_SIZE = CONFIG['batch_size']

# Load Serialized training and validation set
train_set = load_serialized_dataset(CONFIG['dataset']+'_Train',extra_back_steps=1)
val_set = load_serialized_dataset(CONFIG['dataset']+'_Val',extra_back_steps=1)

# Split validation set into validation and test set 
val_set,test_set = train_test_data_split(val_set,TRAIN_TEST_RATIO=0.8)

# Get single dataset example for NN dimensionality
params,fields = get_single_dataset_example(train_set)

#Define model
model = PDNN(input_size = params.shape[1],
            num_layers=CONFIG['num_layers'],
            phi_k = CONFIG['phi_k'],
            s_c = CONFIG['s_c'],
            alpha = CONFIG['alpha'])

print(f'Model Parameters {model.get_number_of_parameters():e}')

if torch.cuda.device_count() > 1:
  print("Multiple GPUs found, using:", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

DROP_LAST = False
PIN_MEMORY = TRUE
NUM_WORKERS = 4
SHUFFLE = False # Data is already shuffled and excessive shuffles slows training 
PERSISTENT_WORKERS = True

# Create dataloaders for training, validation and test datasets
train_dataloader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=SHUFFLE,num_workers = NUM_WORKERS,drop_last=DROP_LAST, pin_memory=PIN_MEMORY,persistent_workers = PERSISTENT_WORKERS)
val_dataloader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle=SHUFFLE,num_workers=NUM_WORKERS,drop_last=DROP_LAST, pin_memory=PIN_MEMORY,persistent_workers = PERSISTENT_WORKERS)
test_dataloader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle=SHUFFLE,num_workers=NUM_WORKERS,drop_last=DROP_LAST, pin_memory=PIN_MEMORY,persistent_workers = PERSISTENT_WORKERS)

model = model.to(device)
print('Model:')
print(model)

criterion = relRMSE_pytorch # Custom loss function for relative RMSE
#criterion = torch.nn.MSELoss() # Runs slightly faster (2-3%) and should ensure same convergence.

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate']) 

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[CONFIG['epochs']*x for x in [0.5,0.6,0.7,0.8,0.9]], verbose = True,gamma = 0.5)

train_loss_array = []
test_loss_array = []
loss_array = []

epoch_times_array = []

train_size = len(train_set)
val_size = len(val_set)
test_size = len(test_set)

best_val_loss = float("inf")
for epoch in range(CONFIG["epochs"]):

    t0 = time.time()
    epoch_training_loss = 0
    epoch_training_targets = 0
    model.train()
    for train_params, train_fields in train_dataloader:
        batch_size = len(train_fields)
        ## Transfer Batch to Device
        train_params = train_params.to(device)
        train_fields = train_fields.to(device)
        
        prediction = model(train_params)
        
        #loss = criterion(input = prediction, target = train_fields)
        loss = criterion(prediction, train_fields)
        epoch_training_loss += loss*(batch_size/train_size)

        optimizer.zero_grad(set_to_none=True) # According to pytorch set_to_none should be faster
        loss.backward()
        optimizer.step()
    
    # Calculate validation evaluation
    model.eval()
    epoch_val_loss = 0
    epoch_val_targets = 0
    with torch.no_grad():
        for val_params, val_fields in val_dataloader:
            batch_size = len(val_fields) # To ensure end batches are also calculated correctly
            # Put thing to device
            val_params = val_params.to(device)
            val_fields = val_fields.to(device)
            
            prediction = model(val_params)
            loss = criterion(input = prediction, target = val_fields)
            epoch_val_loss += loss*(batch_size/val_size)


    # Save model if validation loss is best
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model = model 



    # Make test evaluation
    epoch_test_loss = 0
    epoch_test_targets = 0
    with torch.no_grad():
        model.eval()
        for test_params, test_fields in test_dataloader:
            batch_size = len(test_fields)
            # Put thing to device
            test_params = test_params.to(device)
            test_fields = test_fields.to(device)

            prediction = model(test_params)
            #loss = criterion(input = prediction, target = test_fields)
            loss = criterion(prediction, test_fields)
            epoch_test_loss += loss*(batch_size/test_size)


    scheduler.step() # Step adaptive learning rate scheduler

    t1 = time.time()
    epoch_time = t1-t0
    epoch_times_array.append(epoch_time)

    mean_epoch_rate = 60/(sum(epoch_times_array)/(epoch+1))

    wandb.log({'Train_loss':epoch_training_loss,
                'Val_loss':epoch_val_loss,
                'Test_loss':epoch_test_loss,
                'best_val_loss':best_val_loss,
                'Learning Rate':scheduler.get_last_lr(),
                'epoch_time':epoch_time})
    
    print("epoch : {}/{}, train_loss = {:.9e}, val_loss = {:.9e}, test_loss = {:.9e}, epoch/min: {:.3f}, mean epoch/min {:.3f}".format(epoch + 1, CONFIG["epochs"], loss,epoch_val_loss,epoch_test_loss,60/epoch_time,mean_epoch_rate))
    

saveModel(model=best_model,name = run_name,subfolder=project,extra_step_back=1)

