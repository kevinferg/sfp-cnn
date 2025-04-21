# Scalar Field Prediction on Meshes Using Interpolated Multiresolution Convolutional Neural Networks

![Neural Network Architecture showing a signed distance field being repeatedly convolved and pooled. Each convolution output grid is interpolated back to the initial nodes to make nodal feature vectors. An output MLP then makes node-wise scalar field predictions.](multires-interp-cnn-architecture.png)

#### Authors
_Kevin Ferguson_, Carnegie Mellon University  
James Hardin, Air Force Research Lab  
Andrew Gillman, Air Force Research Lab  
Levent Burak Kara, Carnegie Mellon University  

### Abstract

[Paper Link](https://asmedigitalcollection.asme.org/appliedmechanics/article/91/10/101002/1201208)

Scalar fields, such as stress or temperature fields, are often calculated in shape optimization and design problems in engineering. For complex problems where shapes have varying topology and cannot be parametrized, data-driven scalar field prediction can be faster than traditional finite element methods. However, current data-driven techniques to predict scalar fields are limited to a fixed grid domain, instead of arbitrary mesh structures. In this work, we propose a method to predict scalar fields on arbitrary meshes. It uses a convolutional neural network whose feature maps at multiple resolutions are interpolated to node positions before being fed into a multilayer perceptron to predict solutions to partial differential equations at mesh nodes. The model is trained on finite element von Mises stress fields, and once trained, it can estimate stress values at each node on any input mesh. Two shape datasets are investigated, and the model has strong performance on both, with a median R2 value of 0.91. We also demonstrate the model on a temperature field in a heat conduction problem, where its predictions have a median R2 value of 0.99. Our method provides a potential flexible alternative to finite element analysis in engineering design contexts. A followup study on [Out-of-Distribution Detection](ood/) for the proposed method is also included in the repository.

## Citation

Consider citing the following:  

- Ferguson, K., Gillman, A., Hardin, J., and Kara, L. B. (July 12, 2024). "Scalar Field Prediction on Meshes Using Interpolated Multiresolution Convolutional Neural Networks." ASME. J. Appl. Mech. October 2024; 91(10): 101002. https://doi.org/10.1115/1.4065782

BibTex:
```
@article{ferguson-2024-multiresolution,
    author = {Ferguson, Kevin and Gillman, Andrew and Hardin, James and Kara, Levent Burak},
    title = {Scalar Field Prediction on Meshes Using Interpolated Multiresolution Convolutional Neural Networks},
    journal = {Journal of Applied Mechanics}, volume = {91}, number = {10}, pages = {101002},
    year = {2024}, month = {07},
    issn = {0021-8936},
    doi = {10.1115/1.4065782},
    url = {https://doi.org/10.1115/1.4065782},
    eprint = {https://asmedigitalcollection.asme.org/appliedmechanics/article-pdf/91/10/101002/7353098/jam\_91\_10\_101002.pdf},
}
```

---

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

The `scripts/` folder contains Python scripts that do the following:

- [download_datasets](scripts/download_datasets.py) downloads all 9 datasets from Google Drive and extracts them to the [data/](data/) folder
- [train_models](scripts/train_models.py) creates and trains 11 different models and saves them to the [models/](models/) folder
- [print_tables](scripts/print_tables.py) evaluates trained models on each dataset and prints their R-squared values to a LaTeX table in the [figures/](figures/) folder
- [generate_figures](scripts/generate_figures.py) generates several figures related to model performance and visualization in the [figures/](figures/) folder



### Acknowledgment
This research was funded by Air Force Research Laboratory contract FA8650-21-F-5803.

