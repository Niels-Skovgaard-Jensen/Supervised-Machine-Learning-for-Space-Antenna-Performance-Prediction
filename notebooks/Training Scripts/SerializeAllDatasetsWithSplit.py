
from ssapp.Utils import train_test_data_split
from ssapp.data.AntennaDatasetLoaders import * # I know this is a no-no. But in this particular case i think it is okay


if __name__ == '__main__':
    dataset_constructors = [MLADataset1]

    for dataset_constructor in dataset_constructors:
        print('Next Dataset:',dataset_constructor)
        print('This might take some time...')
        dataset = dataset_constructor()
        print('Dataset loaded:',dataset)
        train_dataset, val_dataset = train_test_data_split(dataset)
        train_dataset.name = dataset.name+'_Train'
        val_dataset.name = dataset.name+'_Val'
        print('Train and Validation Split Names',train_dataset.name, val_dataset.name)
        serialize_dataset(train_dataset)
        serialize_dataset(val_dataset)

        