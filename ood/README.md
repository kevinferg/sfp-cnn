# Out-of-Distribution Detection

We present one method for OOD Detection of field prediction methods:
- Perform PCA on the flattened SDF images
- Compute the PCA log-likelihood of each training SDF
- The 5th percentile training log-likelihood becomes the OOD threshold
- Any new shapes with log-likelihoods below the threshold are rejected

We also leverage the fact that we know how well the field predictor performs on training points by training another model to quantify the field prediction model's performance. There are therefore three distinct models at play:
- Field Predictor: Outputs a field, i.e. a value at every node on a shape
- R-Squared Estimator: Outputs how well the Field Predictor will perform on an input shape
- OOD Detector: Input a shape, output


## Usage

Run the scripts in order:
- [download_ood_data](ood/download_ood_data.py): Download the datasets used for the OOD detection study
- [train_model](ood/train_model.py): Train the field prediction model on the OOD dataset
- [detect_ood](ood/detect_ood.py): Evaluate the R-squared predictor and OOD detector; generate figures
