
from ssapp.Utils import train_test_data_split
from ssapp.data.AntennaDatasetLoaders import serialise_all_datasets, serialize_dataset
from ssapp.data.AntennaDatasetLoaders import PatchAntennaDataset, PatchAntennaDataset2, CircularHornDataset1


if __name__ == '__main__':
    dataset_constructors = [PatchAntennaDataset,
                PatchAntennaDataset2,
                CircularHornDataset1]

    for dataset_constructor in dataset_constructors:
        print(dataset_constructor)
        dataset = dataset_constructor()
        print(dataset)
        train_dataset, val_dataset = train_test_data_split(dataset)
        train_dataset.name = dataset.name+'_Train'
        val_dataset.name = dataset.name+'_Val'
        print(train_dataset.name, val_dataset.name)
        serialize_dataset(train_dataset)
        serialize_dataset(val_dataset)