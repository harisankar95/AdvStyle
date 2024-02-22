import os
from typing import List, Tuple

import cv2
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE


def t_sne(stylefeats: List[np.array]) -> plt.Figure:
    """
    Apply t-SNE for dimensionality reduction and create a scatter plot.

    Parameters
    ----------
    stylefeats : List[np.array]
        The style features of the images.

    Returns
    -------
    fig : plt.Figure
        The figure of the scatter plot.
    """

    # Set the style of the plot
    sns.set_theme()

    # Stack the style features and create the labels
    all_X = np.vstack(stylefeats)
    all_y = np.hstack([np.ones(len(stylefeats[i])) * i for i in range(len(stylefeats))])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=20240220, perplexity=30)
    X_tsne = tsne.fit_transform(all_X)

    # Create a color palette
    palette = np.array(sns.color_palette("hls", len(stylefeats)))

    # Create the scatter plot
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")
    sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], lw=0, s=40, c=palette[all_y.astype(np.int64)])
    ax.axis("tight")

    # Add the labels to the scatter plot
    for i in range(len(stylefeats)):
        # Position of each label.
        xtext, ytext = np.median(X_tsne[all_y == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])

    plt.title("t-SNE of Style Features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.xlim(-80, 80)
    plt.ylim(-80, 80)
    plt.grid(True)

    return fig


def get_means_stds(folder: str) -> Tuple[np.array, np.array]:
    """
    Get the means and standard deviations of the images in the directory.

    Parameters
    ----------
    folder : str
        The directory of the images.

    Returns
    -------
    means : np.array
        The means of the images.
    stds : np.array
        The standard deviations of the images.
    """

    means = []
    stds = []

    # Loop through the images in the directory and get the mean and standard deviation
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(root, file), cv2.IMREAD_COLOR)
                img = img / 255.0
                mean, std = cv2.meanStdDev(img)
                means.append(mean.flatten())
                stds.append(std.flatten())

    means = np.array(means)
    stds = np.array(stds)

    return means, stds


if __name__ == "__main__":

    # Get the means and standard deviations of the images in the test set and synthetic test set
    test_means, test_stds = get_means_stds("./data/test_set")
    synthetic_test_means, synthetic_test_stds = get_means_stds("./data/synthetic_test_set")

    # Apply t-SNE for dimensionality reduction and create a scatter plot
    fig = t_sne([test_means, synthetic_test_means])
    fig.savefig("./assets/t_sne_plot.png")
