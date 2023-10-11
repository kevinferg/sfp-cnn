# Multi-resolution Interpolated CNN

## Requirements

The following Python packages are required:
- NumPy
- SciPy
- Matplotlib
- PyTorch
- Gdown

Run the following commands in Anaconda to create and activate a conda environment with the necessary packages:
```
conda create -y -n multires_env numpy scipy matplotlib pytorch torchvision torchaudio cpuonly -c pytorch
conda activate multires_env
conda install -y -c conda-forge gdown
```

## Usage

- [download_datasets](scripts/download_datasets.py) downloads all 9 datasets from Google Drive and extracts them to the [data/](data/) folder
- [train_models](scripts/train_models.py) creates and trains 11 different models and saves them to the [models/](models/) folder
- [print_tables](scripts/print_tables.py) evaluates trained models on each dataset and prints their R-squared values to a LaTeX table in the [figures/](figures/) folder
- [generate_figures](scripts/generate_figures.py) generates several figures related to model performance and visualization in the [figures/](figures/) folder