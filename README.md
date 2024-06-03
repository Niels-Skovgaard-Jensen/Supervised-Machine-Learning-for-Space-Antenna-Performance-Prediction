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


The project is structured roughly in accordance with the [cookiecutter data science template](https://drivendata.github.io/cookiecutter-data-science/).
```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Most real analysis is done here by running src commands
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├── ssapp              <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
```


The repository includes all of the datasets in raw .txt format, but they are used in serialized form throughout the package and notebooks. To convert all the datasets into serialized form, the SerializeAllDatasets script can be used inside the script.
```
python usr/Supervised-Machine-Learning-for-Space-Antenna-Performance-Prediction\notebooks\Training Scripts\SerializeAllDatasets.py
```
