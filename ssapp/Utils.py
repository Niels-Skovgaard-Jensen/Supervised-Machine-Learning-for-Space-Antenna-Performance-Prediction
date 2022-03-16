from pathlib import Path
from torch.utils.data import random_split



def train_test_data_split(dataset, TRAIN_TEST_RATIO = 0.7):

    train_len = int(len(dataset)*TRAIN_TEST_RATIO)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])    
    return train_set, test_set