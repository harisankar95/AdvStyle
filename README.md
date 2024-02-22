# Adversarial Style for Image Classification

## Introduction

This project is an implementation of the paper "Adversarial Style Augmentation for Domain
Generalized Urban-Scene Segmentation"[[1](https://arxiv.org/abs/2207.04892)] by Z. Zhong et al. The paper proposes a method to improve the performance of a model on unseen real domains by adversarially augmenting the style of the input images. The method is based on the observation that the style of the images can largely influence the model's performance and that the style features can be well represented by the channel-wise mean and standard deviation of the images. The proposed method, called Adversarial Style Augmentation (AdvStyle), regards the style feature as a learnable parameter and updates it by adversarial training. The paper demonstrates that AdvStyle can significantly improve the model performance on unseen real domains for semantic segmentation.

Here, we apply the method to the task of image classification and evaluate its performance kaggel's "Dogs vs. Cats" dataset[[2](https://www.kaggle.com/datasets/tongpython/cat-and-dog/data)]. We create a synthetic test set by applying the style augmentation to the original test set and compare the performance of the model on the original and synthetic test sets.

## Getting Started

### Prerequisites

The code is implemented in Python and requires the following packages:

- PyTorch
- torchvision
- tqdm
- matplotlib
- seaborn

### Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

### Usage

1. Download the "Dogs vs. Cats" dataset from kaggle[[2](https://www.kaggle.com/datasets/tongpython/cat-and-dog/data)] and extract the files to the `data` directory. The directory should have the following structure:

```plaintext
data
├── training_set
│   ├── cats
│   │   ├── cat.1.jpg
│   │   ├── cat.2.jpg
│   │   └── ...
│   └── dogs
│       ├── dog.1.jpg
│       ├── dog.2.jpg
│       └── ...
└── test_set
    ├── cats
    │   ├── cat.4001.jpg
    │   ├── cat.4002.jpg
    │   └── ...
    └── dogs
        ├── dog.4001.jpg
        ├── dog.4002.jpg
        └── ...
```

2. Generate synthetic test sets by running the following command:

```bash
python generate_synthetic_test_set.py
```
