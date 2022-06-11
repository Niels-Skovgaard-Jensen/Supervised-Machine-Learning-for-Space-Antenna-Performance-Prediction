
from ssapp.Utils import train_test_data_split
from ssapp.data.AntennaDatasetLoaders import * # I know this is a no-no. But in this particular case i think it is okay


if __name__ == '__main__':
    dataset_constructors = [MLADataset1,
                            ReflectorCutDataset2,
                            ReflectorCutDataset]

    for dataset_constructor in dataset_constructors:
        print('Next Dataset:',dataset_constructor)
        print('This might take some time...')
        dataset = dataset_constructor()
        print('Dataset loaded:',dataset)
        serialize_dataset(dataset)

