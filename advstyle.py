"""
Adversarial Style Training
"""

from typing import Dict, List, Tuple, Union

import torch
from torch import nn
from tqdm.auto import tqdm


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_adv_style: bool = True,
    adv_lr: int = 1.0,
) -> Tuple[float, float]:
    """
    Train the model for one epoch

    Parameters
    ----------
    model : nn.Module
        Model to train
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer to use for training
    device : str
        Device to use for training
    use_adv_style : bool, optional
        Whether to use adversarial style training, by default True
    adv_lr : int, optional
        Learning rate for adversarial style training, by default 1.0

    Returns
    -------
    Tuple[float, float]
        Average loss and accuracy for the epoch
    """
    # Initialize the running loss and accuracy
    running_loss, running_acc = 0.0, 0.0

    # Set the model to training mode
    model.train()
    for inputs, labels in train_loader:
        # Move the inputs and labels to the device
        inputs, labels = inputs.to(device), labels.to(device)

        if use_adv_style:
            # Compute the mean and standard deviation of the mini-batch
            mu = inputs.mean(dim=[2, 3], keepdim=True)
            std = inputs.std(dim=[2, 3], keepdim=True) + 1e-6  # Add a small value to avoid division by zero

            # Normalize the inputs
            inputs_normed = (inputs - mu) / std
            # Detach the normalized inputs
            inputs_normed = inputs_normed.clone().detach()

            # Create the adversarial mean and standard deviation
            # and set them to require gradients for adversarial style training
            adv_mu = mu.clone().detach().requires_grad_(True)
            adv_std = std.clone().detach().requires_grad_(True)

            # Initialize the adversarial optimizer
            adv_optim = torch.optim.SGD(params=[adv_mu, adv_std], lr=adv_lr, momentum=0, weight_decay=0)

            # Optimize the adversarial mean and standard deviation (style features)
            adv_optim.zero_grad()  # Zero the parameter gradients
            inputs_adv = inputs_normed * adv_std + adv_mu  # Denormalize the inputs
            outputs_adv = model(inputs_adv)  # Forward pass
            loss_adv = criterion(outputs_adv, labels)  # Calculate the loss
            (-loss_adv).backward()  # Backward pass (maximize the loss)
            adv_optim.step()  # Update the adversarial mean and standard deviation

            # Detach the adversarial mean and standard deviation
            # to prevent the gradients from being propagated to the style features
            adv_mu = adv_mu.clone().detach()
            adv_std = adv_std.clone().detach()

            # Denormalize the inputs using the optimized adversarial mean and standard deviation
            inputs_adv = inputs_normed * adv_std + adv_mu
            inputs = torch.cat([inputs, inputs_adv], dim=0)  # Concatenate the original and adversarial inputs
            labels = torch.cat([labels, labels], dim=0)  # Concatenate the original and adversarial labels

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights

        running_loss += loss.item()

        # Calculate the accuracy
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            running_acc += (preds == labels).float().mean().item()

    epoch_loss = running_loss / len(train_loader)  # Calculate the average loss for the epoch
    epoch_acc = running_acc / len(train_loader)  # Calculate the average accuracy for the epoch

    return epoch_loss, epoch_acc


def test_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    synthetic_test_loader: Union[torch.utils.data.DataLoader, None] = None,
) -> Tuple[float, float, float]:
    """
    Test the model

    Parameters
    ----------
    model : nn.Module
        Model to test
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test set
    criterion : nn.Module
        Loss function
    device : str
        Device to use for testing
    synthetic_test_loader : Union[torch.utils.data.DataLoader, None], optional
        DataLoader for the synthetic test set, by default None

    Returns
    -------
    Tuple[float, float, float]
        Average loss, accuracy, and synthetic test accuracy
    """

    # Initialize the running loss and accuracy
    running_loss, running_acc, running_synthetic_acc = 0.0, 0.0, 0.0

    # Set the model to evaluation mode
    model.eval()

    # Test the model on the test set
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move the inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate the accuracy
            preds = torch.argmax(outputs, dim=1)
            running_acc += (preds == labels).float().mean().item()

        # Calculate the average loss and accuracy for the test set
        test_loss = running_loss / len(test_loader)
        test_acc = running_acc / len(test_loader)

        # Test the model on the synthetic test set if provided
        if synthetic_test_loader is not None:
            for inputs, labels in synthetic_test_loader:
                # Move the inputs and labels to the device
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate the accuracy
                preds = torch.argmax(outputs, dim=1)
                running_synthetic_acc += (preds == labels).float().mean().item()

            # Calculate the average synthetic test accuracy
            synthetic_test_acc = running_synthetic_acc / len(synthetic_test_loader)
        else:
            synthetic_test_acc = 0.0

    return test_loss, test_acc, synthetic_test_acc


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int,
    use_adv_style: bool = True,
    adv_lr: int = 1.0,
    synthetic_test_loader: Union[torch.utils.data.DataLoader, None] = None,
) -> Dict[str, List]:
    """
    Train the model

    Parameters
    ----------
    model : nn.Module
        Model to train
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test set
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer to use for training
    device : str
        Device to use for training
    num_epochs : int
        Number of epochs to train the model
    use_adv_style : bool, optional
        Whether to use adversarial style training, by default True
    adv_lr : int, optional
        Learning rate for adversarial style training, by default 1.0
    synthetic_test_loader : Union[torch.utils.data.DataLoader, None], optional
        DataLoader for the synthetic test set, by default None

    Returns
    -------
    Dict[str, List]
        Dictionary containing the training and test results
    """

    # Initialize the results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "synthetic_test_acc": [],
    }

    # Initialize the progress bar
    pbar = tqdm(total=num_epochs, desc="Training Progress")
    for epoch in range(num_epochs):
        # Train the model for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_adv_style, adv_lr)

        # Test the model
        test_loss, test_acc, synthetic_test_acc = test_model(
            model, test_loader, criterion, device, synthetic_test_loader
        )

        # Update progress bar
        pbar.update(1)

        # Update results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["synthetic_test_acc"].append(synthetic_test_acc)

        # Print the loss and accuracy for the epoch
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
            f"Synthetic Test Acc: {synthetic_test_acc:.4f}"
        )

    # Close the progress bar
    pbar.close()

    return results
