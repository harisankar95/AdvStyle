""" Utility functions for the project """

from typing import Dict, List, Union

import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_acc(results: Dict[str, List]) -> plt.Figure:
    """
    Plot the training and test loss and accuracy

    Parameters
    ----------
    results : Dict[str, List]
        Dictionary containing the training and test results

    Returns
    -------
    plt.Figure
        Figure containing the plots
    """
    # Set default style
    sns.set_theme()

    # Create a new figure
    fig = plt.figure(figsize=(12, 6))

    # Plot the training and test loss
    plt.subplot(1, 2, 1)
    sns.lineplot(x=range(len(results["train_loss"])), y=results["train_loss"], label="train")
    sns.lineplot(x=range(len(results["test_loss"])), y=results["test_loss"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()

    # Plot the training and test accuracy
    plt.subplot(1, 2, 2)
    sns.lineplot(x=range(len(results["train_acc"])), y=results["train_acc"], label="train")
    sns.lineplot(x=range(len(results["test_acc"])), y=results["test_acc"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.tight_layout()

    # Return the figure
    return fig


def plot_key(
    results: List[Dict[str, Union[float, int]]], key: str, title: str, labels: Union[List[str], None] = None
) -> plt.Figure:
    """
    Plot a key from the results

    Parameters
    ----------
    results : List[Dict[str, Union[float, int]]]
        List of dictionaries containing the results
    key : str
        Key to plot
    title : str
        Title of the plot
    labels : Union[List[str], None], optional
        List of labels for the plot, by default None

    Returns
    -------
    plt.Figure
        Figure containing the plot

    Raises
    ------
    ValueError
        - If the results are not a list of dictionaries
        - If the key is not in the dictionaries
        - If the labels have a different length than the results
    """
    # Check if results is a list of dictionaries
    if not all(isinstance(result, dict) for result in results):
        raise ValueError("Results must be a list of dictionaries")

    # Check if the key is in the dictionaries
    if not all(key in result for result in results):
        raise ValueError("Key must be in the dictionaries")

    # Set default style
    sns.set_theme()

    # Create a new figure
    fig = plt.figure(figsize=(12, 6))

    # Create labels if not provided
    if labels is None:
        labels = [f"run {i + 1}" for i in range(len(results))]
    elif len(labels) != len(results):
        raise ValueError("Labels must have the same length as results")

    # Plot the key
    for i, result in enumerate(results):
        sns.lineplot(x=range(len(result[key])), y=result[key], label=labels[i])
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Return the figure
    return fig
