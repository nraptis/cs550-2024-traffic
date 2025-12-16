import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from pathlib import Path

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    if images:
        min_width = images[0].shape[0]
        min_height = images[0].shape[1]
        max_width = images[0].shape[0]
        max_height = images[0].shape[1]
        for image in images:
            width = images[0].shape[0]
            height = images[0].shape[1]
            min_width = min(min_width, width)
            max_width = max(max_width, width)
            min_height = min(min_height, height)
            max_height = max(max_height, height)
        print("min_width = ", min_width)
        print("min_height = ", min_height)
        print("max_width = ", max_width)
        print("min_hmax_heighteight = ", max_height)
        
        
        

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    project_root = Path(__file__).resolve().parent
    data_root = project_root / Path(data_dir)
    print("data_root = ", data_root)
    print("exists:", data_root.exists())
    print("is_dir:", data_root.is_dir())

    images = []
    labels = []
    for category_dir in sorted(data_root.iterdir()):
        if not category_dir.is_dir():
            continue
        label = int(category_dir.name)
        for img_path in category_dir.iterdir():
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(label) 
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """



if __name__ == "__main__":
    main()
