""" 
This script is used to generate a synthetic test set by augmenting the images in the test set. 
The images are augmented by changing the mean and standard deviation. 
The augmented images are saved in a new folder called synthetic_test_set. 
"""

import os
from typing import List

import cv2
import numpy as np
from tqdm.auto import tqdm

# Artifically augment the mean and std to make the images look like synthetic images
NEW_MEAN = [0.79, 0.59, 0.21]
NEW_STD = [0.23, 0.23, 0.23]


# Augment the images and save them in a new folder called synthetic_test_set
def augment_images(folder: str, new_mean: List, new_std: List):
    """
    Augment the images in the test set by changing the mean and standard deviation.

    Parameters
    ----------
    folder : str
        The directory of the test set.
    new_mean : List
        The new mean to use for the images.
    new_std : List
        The new standard deviation to use for the images.
    """
    # The mean and standard deviation of the current images
    current_mean = [0.413, 0.452, 0.486]
    current_std = [0.224, 0.224, 0.230]

    # Create the new folder if it does not exist
    os.makedirs(folder.replace("test_set", "synthetic_test_set"), exist_ok=True)

    # Loop through the images in the test set
    for root, _, files in os.walk(folder):
        for file in tqdm(files, desc="Augmenting images", total=len(files)):
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(root, file), cv2.IMREAD_COLOR)
                img = img / 255.0
                img = (img - current_mean) / current_std * new_std + new_mean
                img = np.clip(img, 0, 1)
                img = (img * 255.0).astype(np.uint8)
                cv2.imwrite(os.path.join(root, file).replace("test_set", "synthetic_test_set"), img)


if __name__ == "__main__":

    # Augment the images in the test set
    augment_images("./data/test_set", NEW_MEAN, NEW_STD)
