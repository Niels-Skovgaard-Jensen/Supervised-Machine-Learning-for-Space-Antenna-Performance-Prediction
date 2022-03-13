# Standard Packages
from torch.utils.data import random_split, DataLoader
import torch
import wandb
from datetime import datetime
from matplotlib import pyplot as plt
# Custom Packages
from NeuralNetworkModels.SimpleFeedForward import DirectFeedForwardNet
from AntennaDatasets import AntennaDatasetLoaders

# General Settings
torch.manual_seed(42) # Manual seed for sanity
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

wandb.init(project="my-test-project", entity="skoogy_dan")




CUTS_IN_DATASET = 12

dataset = AntennaDatasetLoaders.ReflectorCutDataset(cuts = CUTS_IN_DATASET,flatten_output = True)

TEST_TRAIN_RATIO = 0.8
BATCH_SIZE = 1
train_len = int(len(dataset)*TEST_TRAIN_RATIO)
train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

train_dataloader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle=True)



#Define model
model = DirectFeedForwardNet(in_features = 3,out_features = 4004,NN = 40)
model = model.to(device)
#Test Forward pass
print(model(torch.randn(2,3).to(device)))

LEARNING_RATE = 4e-4

criterion = torch.nn.MSELoss()



EPOCHS = int(2e3)


wandb.config = {
  "learning_rate": LEARNING_RATE,
  "epochs": EPOCHS,
  "batch_size": BATCH_SIZE
}


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

startTime = datetime.now()
train_loss_array = []
test_loss_array = []
loss_array = []

for epoch in range(EPOCHS):

    epoch_training_loss = 0
    epoch_training_targets = 0
    train_batches = 0
    for input_train_batch, target_train_batch in train_dataloader:
        
        ## Transfer Batch to Device
        input_train_batch = input_train_batch.to(device)
        target_train_batch = target_train_batch.to(device)
        
        prediction = model(input_train_batch)
        
        loss = criterion(input = prediction, target = target_train_batch)
        epoch_training_loss += loss/(target_train_batch.shape[0]*target_train_batch.shape[1])
        train_batches +=1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
    
    epoch_test_loss = 0
    epoch_test_targets = 0
    test_batches=0
    with torch.no_grad():
        
        for input_test_batch, target_test_batch in test_dataloader:
            
            # Put thing to device
            input_test_batch = input_test_batch.to(device)
            target_test_batch = target_test_batch.to(device)
            
            
            prediction = model(input_train_batch)
            loss = criterion(input = prediction, target = target_test_batch)
            epoch_test_loss += loss/(target_test_batch.shape[0]*target_test_batch.shape[1])
            test_batches += 1
    
    if epoch%(EPOCHS/10) == 0:
        print('Training Loss',(epoch_training_loss/train_batches).item())
        print('Test Loss',(epoch_test_loss/test_batches).item())

    wandb.log({"Training Loss": epoch_training_loss,
                "Test Loss" : epoch_test_loss})





print('Training time:', datetime.now()-startTime)