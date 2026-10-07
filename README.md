# Brain Tumor MRI Classification with PyTorch

A simple CNN-based project for classifying brain MRI images into four classes using **PyTorch**.

## About Brain Tumors

A brain tumor is an abnormal growth of cells in or around the brain. Brain tumors can have different types and characteristics, so identifying their type is an important part of medical evaluation.

MRI scans provide detailed images of the brain and can contain visual patterns that help distinguish between different tumor types.

## Project Goal

The goal of this project is to build a deep learning model that can automatically classify brain MRI images into four categories:

* Glioma
* Meningioma
* No Tumor
* Pituitary

This project is mainly focused on learning how to build an image classification pipeline with PyTorch. It is **not intended for medical diagnosis**.

## What this project includes

* Image preprocessing and augmentation
* PyTorch `Dataset` and `DataLoader`
* CNN model
* Model training with Adam
* Learning-rate scheduling
* Best model checkpointing
* Accuracy, Precision, Recall, and F1-score
* Confusion matrix
* Sample predictions and misclassified images

## Model

The CNN uses four convolutional layers followed by pooling and fully connected layers.

## Dataset

The project uses the Brain Tumor MRI Dataset with separate `Training` and `Testing` folders.

Update the dataset path before running:

```python
DATA_ROOT = Path("/kaggle/input/brain-tumor-mri-dataset")
```

## Requirements

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow tqdm
```

Run the notebook from top to bottom.

