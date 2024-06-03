# Supervised machine learning for Space Antenna Performance Prediction (ssapp)
### By Niels Skovgaard Jensen
This repository is an overview of the work conducted as my Master's Thesis at The Technical University of Denmark in collaboration with TICRA FOND. 

The main objective is to create surrogate antenna models from data created by fine antenna models. Here, the TICRAS Toolset has been used for generating far-field antenna patterns of different antenna configurations, which are then modelled by a set of different machine learning methods.

For an overview and more stuff look [here](https://skoogydan.github.io/Supervised-Machine-Learning-for-Space-Antenna-Performance-Prediction/)

The easiest way to install is simply to make a new virtual environment with python 3.8.11 and run

```
python install -r requirements.txt
```

If you just want to use the package
```
python setup.py install
```
If you want to edit in the package:
```
python setup.py develop
```

The repository includes all of the datasets in raw .txt format, but they are used in serialized form throughout the package and notebooks. To convert all the datasets into serialized form, the SerializeAllDatasets script can be used inside the script.
```
python usr/Supervised-Machine-Learning-for-Space-Antenna-Performance-Prediction\notebooks\Training Scripts\SerializeAllDatasets.py
```
