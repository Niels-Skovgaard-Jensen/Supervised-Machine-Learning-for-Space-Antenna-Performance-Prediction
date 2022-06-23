
from ssapp.Utils import train_test_data_split
from ssapp.data.AntennaDatasetLoaders import * # I know this is a no-no. But in this particular case i think it is okay
import yaml

WITH_SPLIT = True

if __name__ == '__main__':




    dataset_constructors = [
                            MLADataset1,
                            #ReflectorAntennaDataset3,
                            #PatchAntennaDataset2,
                            #CircularHornDataset1
                            ]

    print('Running')
    print('WITH_SPLIT:', WITH_SPLIT)

    for dataset_constructor in dataset_constructors:
        print('Next Dataset:',dataset_constructor)
        print('This might take some time...')
        dataset = dataset_constructor()
        print('Dataset loaded:',dataset)


        if WITH_SPLIT:
            train_dataset, val_dataset = train_test_data_split(dataset)
            train_dataset.name = dataset.name+'_Train'
            val_dataset.name = dataset.name+'_Val'
            serialize_dataset(train_dataset)
            serialize_dataset(val_dataset)
        elif WITH_SPLIT == 'Both':
            train_dataset, val_dataset = train_test_data_split(dataset)
            train_dataset.name = dataset.name+'_Train'
            val_dataset.name = dataset.name+'_Val'
            serialize_dataset(train_dataset)
            serialize_dataset(val_dataset)
            serialize_dataset(dataset)
        else:
            serialize_dataset(dataset)

        

        