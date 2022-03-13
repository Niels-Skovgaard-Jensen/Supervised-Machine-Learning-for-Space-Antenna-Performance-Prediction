import torch
from pathlib import Path

def TrainNetwork(model, epochs, optimizer, criterion, train_dataloader, test_dataloader):
    
    train_loss_array = []
    test_loss_array = []
    total_loss_array = []
    
    for epoch in range(epochs):

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
            
    return model, train_loss_array, test_loss_array



def saveModel(model,name):



    torch.save(model.state_dict(), PATH)


    return True

def loadModel(PATH,modelClass):
    model = modelClass()
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return True