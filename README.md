# mango-weight-estimation
Image-based mango weight estimation project using transfer learning and regression.

# Overview

This project explores whether the weight of Alphonso mangoes can be predicted from images using a pretrained CNN regression model.

The workflow was designed to avoid data leakage by splitting data at the sample level rather than the image level, because each mango has two images. A numeric-feature baseline was also built for comparison. The dataset consists of 71 mango samples and 142 images in total.

# Dataset
Dataset: Alphonso Mangoes Image Dataset
Each mango sample has:
2 images
fruit diameter
shoulder width
actual weight
Total:
71 samples
142 images

In this project, the split was performed using sample IDs, so the two images of the same mango were always assigned to the same fold.

# Project Goal

The goal of this project was not only to train a CNN model, but also to compare different approaches for a small image regression dataset.

Main questions:

Can mango weight be estimated from images alone?
How does an image-based CNN compare with a structured-data baseline?
What does the model attend to when predicting weight?

# Methods
Baseline

A baseline regression model was built using only structured features:

fruit diameter
shoulder width

Tested models:

Linear Regression
Ridge
Random Forest Regressor

Among them, Random Forest performed best with approximately:

MAE: 8.43
RMSE: 10.56
R²: 0.860
CNN Regression

For the image-based model:

Backbone: ResNet18
Initialization: ImageNet pretrained
Training strategy: full fine-tuning
Validation: 5-fold sample-level cross-validation

At first, image-only regression was unstable. After applying target scaling to the weight values, performance improved substantially. The final full fine-tuning result was approximately:

MAE: 23.52 g
RMSE: 27.90 g
R²: 0.158

# Key Findings
Structured numeric features outperformed the image-only CNN on this small dataset. The Random Forest baseline was clearly stronger than the ResNet18 regression model.
Target scaling was important for stabilizing CNN regression training. Before scaling, validation error was extremely high; after scaling, performance improved a lot.
Grad-CAM suggested that the CNN did not fully rely on the mango body itself. The attention maps appeared to react not only to the fruit region but also to background or boundary patterns, which may explain the limited performance. This is an interpretation based on the observed visualizations.


# Why this project matters

This project is meaningful as a portfolio project because it does more than simply train a model.

It includes:

leakage-aware validation design
baseline vs deep learning comparison
regression-specific training adjustments
interpretability with Grad-CAM
analysis of why a CNN may fail on a small dataset

So even though the CNN did not beat the baseline, the project shows practical experimentation and model evaluation rather than just implementation.

# Repository Structure
mango-weight-estimation/
├── baseline.py
├── dataset.py
├── model.py
├── train.py
├── gradcam_utils.py
├── checkpoints/
├── head_only_results.csv
├── full_finetune_results.csv
└── README.md
# How to Run
Baseline
python baseline.py
CNN training
python train.py
Grad-CAM visualization
python gradcam_utils.py

# Future Improvements

Possible next steps:

better background removal or object-centered cropping
multimodal model using both images and numeric features
additional regularization and learning-rate tuning
more data for image-based regression

# One-line Summary

A small-scale regression project that compares structured-feature baselines and image-based deep learning for mango weight estimation, while emphasizing leakage-free validation and model interpretability.
