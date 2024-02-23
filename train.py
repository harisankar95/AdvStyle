import gc
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.advstyle import train_model
from src.model import SimpleImageClassifier
from src.utils import plot_key, plot_loss_acc

# Define the directories for the training and test sets
TRAIN_DIR = "./data/training_set"
TEST_DIR = "./data/test_set"
SYNTHETIC_TEST_DIR = "./data/synthetic_test_set"

# Define the image sizes
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

# Dataloader parameters
BATCH_SIZE = 16
NUM_WORKERS = 4 if os.cpu_count() > 4 else os.cpu_count()

# Training parameters
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9


if __name__ == "__main__":

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    # Define the transforms for the training and test sets
    # The training set uses the TrivialAugmentWide transformation
    # and the test set does not
    train_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),  # Resize the image to the desired size
            transforms.TrivialAugmentWide(),  # Apply the TrivialAugmentWide transformation
            transforms.ToTensor(),  # Convert the image to a tensor
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )

    # Load the training and test sets
    train_set = datasets.ImageFolder(
        TRAIN_DIR,  # Directory of the training set
        transform=train_transform,  # Transform to apply to the images
    )
    test_set = datasets.ImageFolder(
        TEST_DIR,  # Directory of the test set
        transform=test_transform,
    )
    synthetic_test_set = datasets.ImageFolder(
        SYNTHETIC_TEST_DIR,
        transform=test_transform,
    )

    # Create the data loaders
    train_loader = DataLoader(
        train_set,  # Training dataset
        batch_size=BATCH_SIZE,  # Batch size
        shuffle=True,  # Shuffle the data
        num_workers=NUM_WORKERS,  # Number of workers for loading the data
    )
    test_loader = DataLoader(
        test_set,  # Test dataset
        batch_size=BATCH_SIZE,
        shuffle=False,  # Do not shuffle the data
        num_workers=NUM_WORKERS,
    )
    synthetic_test_loader = DataLoader(
        synthetic_test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Do not shuffle the data
        num_workers=NUM_WORKERS,
    )

    # Initialize the model and optimizer (SGD with momentum and weight decay)
    model = SimpleImageClassifier(num_classes=2).to(device)
    loss_fn = nn.CrossEntropyLoss()  # Use cross-entropy loss
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

    # Train the model without adversarial style training
    results = train_model(
        model,
        train_loader,
        test_loader,
        loss_fn,
        optimizer,
        device,
        NUM_EPOCHS,
        use_adv_style=False,
        synthetic_test_loader=synthetic_test_loader,
    )
    # Plot the training and test loss and accuracy
    fig = plot_loss_acc(results)
    fig.savefig("./results/without_advstyle_results.png")

    # Delete the model and optimizer
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize a new model and optimizer
    model = SimpleImageClassifier(num_classes=2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

    # Train the model with adversarial style training
    advstyle_results = train_model(
        model,
        train_loader,
        test_loader,
        loss_fn,
        optimizer,
        device,
        NUM_EPOCHS,
        use_adv_style=True,
        adv_lr=1.0,
        synthetic_test_loader=synthetic_test_loader,
    )
    # Plot the training and test loss and accuracy
    fig = plot_loss_acc(advstyle_results)
    fig.savefig("./results/advstyle_results.png")

    # Plot the accuracy of the synthetic test set
    fig = plot_key(
        [results, advstyle_results],
        "synthetic_test_acc",
        "Synthetic Test Accuracy",
        labels=["Without AdvStyle", "With AdvStyle"],
    )
    fig.savefig("./results/synthetic_test_acc.png")
