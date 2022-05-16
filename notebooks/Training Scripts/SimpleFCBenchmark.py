# Standard Packages
from torch.utils.data import random_split, DataLoader
import torch
import wandb
from datetime import datetime
from matplotlib import pyplot as plt
import torch.nn as nn
# Custom Packages
from ssapp.data.AntennaDatasetLoaders import load_serialized_dataset
from ssapp.Utils import train_test_data_split
from ssapp.data.Metrics import relRMSE, relRMSE_pytorch
from ssapp.models.HelperFunctions import saveModel
from ssapp.models.NeuralNetworkModels.SimpleFeedForward import FCBenchmark

# General Settings
torch.manual_seed(42) # Manual seed for sanity
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





#dataset = load_serialized_dataset('CircularHornDataset1')

BATCH_SIZE = 16
NUM_WORKERS = 4

train_set = load_serialized_dataset('PatchAntennaDataset2_Train')
test_set = load_serialized_dataset('PatchAntennaDataset2_Val')

#train_set = load_serialized_dataset('CircularHornDataset1_Train')
#test_set = load_serialized_dataset('CircularHornDataset1_Val')

train_dataloader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True,num_workers = NUM_WORKERS,drop_last=True)
test_dataloader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS,drop_last=True)

params,fields = next(iter(train_dataloader))
print(params.shape)


#Define model
model = FCBenchmark(input_size = params.shape[1])

model = model.to(device)
#Test Forward pass


LEARNING_RATE = 3e-4

criterion = torch.nn.MSELoss()

EPOCHS = int(800)




optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,200,300,400,500,600,700], verbose = True)


startTime = datetime.now()
train_loss_array = []
test_loss_array = []
loss_array = []

train_size = len(train_set)
test_size = len(test_set)
best_val_loss = float("inf")
for epoch in range(EPOCHS):

    epoch_training_loss = 0
    epoch_training_targets = 0
    train_batches = 0
    model.train()
    for train_params, train_fields in train_dataloader:
        batch_size = len(train_fields)
        ## Transfer Batch to Device
        train_params = train_params.float().to(device)
        train_fields = train_fields.float().to(device)
        
        

        prediction = model(train_params)
        

        #loss = criterion(input = prediction, target = train_fields)
        loss = relRMSE_pytorch(prediction, train_fields)
        epoch_training_loss += loss*(batch_size/train_size)
        train_batches +=1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
    
    epoch_test_loss = 0
    epoch_test_targets = 0
    test_batches=0
    with torch.no_grad():
        model.eval()
        for test_params, test_fields in test_dataloader:
            batch_size = len(test_fields) # To ensure end batches are also calculated correctly
            # Put thing to device
            test_params = test_params.float().to(device)
            test_fields = test_fields.float().to(device)
            
            
            
            prediction = model(test_params)
            #loss = criterion(input = prediction, target = test_fields)
            loss = relRMSE_pytorch(prediction, test_fields)
            epoch_test_loss += loss*(batch_size/test_size)
            


            test_batches += 1
    
    if epoch_test_loss < best_val_loss:
        best_val_loss = epoch_test_loss
        saveModel(model=model,name = 'Simple_Feed_Forward_Patch')
        
    scheduler.step()
    print("epoch : {}/{}, train_loss = {:.9e}, val_loss = {:.9e}".format(epoch + 1, EPOCHS, loss,epoch_test_loss))
    

print('Training time:', datetime.now()-startTime)