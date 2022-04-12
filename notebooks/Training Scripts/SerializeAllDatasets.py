
from ssapp.data.AntennaDatasetLoaders import serialise_all_datasets, serialize_dataset
from ssapp.data.AntennaDatasetLoaders import PatchAntennaDataset, PatchAntennaDataset2, CircularHornDataset1


if __name__ == '__main__':
    datasets = [PatchAntennaDataset(),
                PatchAntennaDataset2(),
                CircularHornDataset1()]

    for dataset in datasets:
        print(dataset)
        serialize_dataset(dataset)