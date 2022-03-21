from genericpath import exists
import torch
from torch.utils.data.dataloader import DataLoader
import wandb


def train(model : torch.nn, 
        config, 
        train_dataloader: DataLoader,
        test_dataloader: DataLoader, 
        optimizer,
        criterion):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCHS = config['epochs']
    BATCH_SIZE = config['batch_size']

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
            # compute reconstructions
            train_outputs = model(field)
                
            # compute training reconstruction loss
            train_loss = criterion(train_outputs, field)
                
            # compute accumulated gradients
            train_loss.backward()
                
            # perform parameter update based on current gradients
            optimizer.step()
                
            # add the mini-batch training loss to epoch loss
            loss += train_loss.detach().item()

        with torch.no_grad():
            for params,field in test_dataloader:
                field = field.float().to(device)
                
                val_outputs = model(field)
                
                test_loss += criterion(val_outputs, field)
                
        
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