# DASH
This is the online repository that stores code, data and supplementary figures of the research work: <Leveraging Data Mining, Active Learning, and Domain Adaptation in a Multi-Stage, Machine Learning-Driven Approach for the Efficient Discovery of Advanced Acidic Oxygen Evolution Electrocatalysts> by Rui Ding et. al. (https://arxiv.org/abs/2407.04877)
![Workflow Schematic](https://github.com/ruiding-uchicago/D.A.S.H./blob/main/Online%20Repository%20Figures/D.A.S.H.%20Workflow.jpg)
## Code Description

Our code is written and uploaded on GitHub in Jupyter notebook file format, which is already very easy to conduct and get results. Our code does not include providing a new python library, but users could track how we train machine learning models in Jupyter ipynb files. Also, we have provided all original dataset CSV files to drive the active learning flow. Additionally, we have detailed the python version and package environment used for this work, focusing on machine learning-related packages.

## Environment
This project was trained using Python version 3.9.12. Below is the list of machine learning-related packages and their versions used in this project on the local machine we used equipped with 2080 Ti GPU, users could install anaconda (https://www.anaconda.com/download) and using pip install to install the packages below for setting environment to run in the Jupyter notebook (https://docs.anaconda.com/ae-notebooks/user-guide/basic-tasks/apps/jupyter/index.html):

absl-py 1.2.0
ase 3.22.1
catboost 1.0.6
cython 0.29.28
dask 2022.2.1
deep-forest 0.1.7
gensim 4.1.2
h5py 3.6.0
keras 2.9.0
Keras-Preprocessing 1.1.2
keras-tcn 3.5.0
lightgbm 3.3.2
matplotlib 3.4.3
numpy 1.23.5
pandas 1.4.2
pytorch-lightning 2.0.1
pymatgen 2023.8.10
scikit-learn 1.0.2
scipy 1.8.0
scikit-opt  0.6.6
scikit-optimize 0.9.0
seaborn 0.11.2
tensorboard                  2.10.1
tensorboard-data-server      0.6.1
tensorboard-plugin-wit       1.8.1
tensorflow-estimator         2.10.0
tensorflow-gpu               2.10.0
tensorflow-io-gcs-filesystem 0.31.0
torch 2.0.0
torchvision 0.15.1
xgboost 1.6.2

