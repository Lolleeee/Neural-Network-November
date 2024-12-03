import numpy as np
import random

def split_set(data, shuffle=True, size=0.2, count=-1, starting=-1):
    """
    Given a balanced dataset, using `split_set(augmented_data)`
    will return two datasets, e.g. one for training and one with validation.
    Generally, if a set contains an augmented image, it contains
    also its original version, and viceversa.

    The argument `size` can be used to set the percentage of the
    dataset to reserve for the first set (e.g. `size=0.2`).

    The argument `count` can be used to set the number of images of the
    dataset to reserve for the first set (e.g. `count=200`).
    If `count` is set, `size` is ignored.

    The argument `shuffle` can be used to shuffle the two returned
    sets, as otherwise the classes are found in blocks one after
    the other (e.g. `shuffle=True`).

    The argument `starting` can be used to set the starting point
    (represented as a percentage) from which to start picking
    images for the first set (e.g. `starting=0.6`).
    """
    # Retrieve images and labels from augmented dataset
    images = data['images']
    labels = data['labels'].flatten()

    # Select starting point from which to pick images
    max_count = np.min(np.bincount(labels))
    if starting == -1:
        start = random.randint(0, max_count - 1)
    else:
        start = int(max_count * starting)

    # Define sizes (names assume validation and training split)
    if count <= 0:
        val_size = int(max_count * size)
    else:
        val_size = count
    classes = len(np.unique(labels))

    # Initialize lists for training and validation sets
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    # Get indices for each label (indices[label][index])
    indices = [np.where(labels.flatten() == i)[0] for i in range(classes)]

    # Select consecutive images for each label
    for label in range(classes):
        max_count = np.count_nonzero(labels == label)

        # Pick images for validation set
        for i in range(val_size):
            current_index = indices[label][(start + i) % max_count]
            val_images.append(images[current_index])
            val_labels.append(label)

        # Pick images for training set
        for i in range(max_count - val_size):
            current_index = indices[label][(start + val_size + i) % max_count]
            train_images.append(images[current_index])
            train_labels.append(label)

    # Shuffle the sets
    if shuffle:
        shuffle_indices = np.random.permutation(len(val_images))
        val_images = [val_images[i] for i in shuffle_indices]
        val_labels = [val_labels[i] for i in shuffle_indices]
        shuffle_indices = np.random.permutation(len(train_images))
        train_images = [train_images[i] for i in shuffle_indices]
        train_labels = [train_labels[i] for i in shuffle_indices]

    # Convert to numpy arrays for saving
    val_images = np.array(val_images)
    val_labels = np.array(val_labels, dtype=np.uint8).reshape(-1, 1)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels, dtype=np.uint8).reshape(-1, 1)

    # Return a dictionary with the same structure as input data
    return {'images': val_images, 'labels': val_labels}, {'images': train_images, 'labels': train_labels}

def split_masked_set(data, shuffle=True, size=0.2, count=-1, starting=-1):
    """
    Given a balanced dataset, using `split_set(augmented_data)`
    will return two datasets, e.g. one for training and one with validation.
    Generally, if a set contains an augmented image, it contains
    also its original version, and viceversa.

    The argument `size` can be used to set the percentage of the
    dataset to reserve for the first set (e.g. `size=0.2`).

    The argument `count` can be used to set the number of images of the
    dataset to reserve for the first set (e.g. `count=200`).
    If `count` is set, `size` is ignored.

    The argument `shuffle` can be used to shuffle the two returned
    sets, as otherwise the classes are found in blocks one after
    the other (e.g. `shuffle=True`).

    The argument `starting` can be used to set the starting point
    (represented as a percentage) from which to start picking
    images for the first set (e.g. `starting=0.6`).
    """
    # Retrieve images and labels from augmented dataset
    images = data['images']
    labels = data['labels']

    # Select starting point from which to pick images
    max_count = len(labels)
    if starting == -1:
        start = random.randint(0, max_count - 1)
    else:
        start = int(max_count * starting)

    # Define sizes (names assume validation and training split)
    if count <= 0:
        val_size = int(max_count * size)
    else:
        val_size = count

    # Initialize lists for training and validation sets
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    # Pick images for validation set
    for i in range(val_size):
        val_images.append(images[(start + i) % max_count])
        val_labels.append(labels[(start + i) % max_count])

    # Pick images for training set
    for i in range(max_count - val_size):
        train_images.append(images[(start + val_size + i) % max_count])
        train_labels.append(labels[(start + val_size + i) % max_count])

    # Shuffle the sets
    if shuffle:
        shuffle_indices = np.random.permutation(len(val_images))
        val_images = [val_images[i] for i in shuffle_indices]
        val_labels = [val_labels[i] for i in shuffle_indices]
        shuffle_indices = np.random.permutation(len(train_images))
        train_images = [train_images[i] for i in shuffle_indices]
        train_labels = [train_labels[i] for i in shuffle_indices]

    # Convert to numpy arrays for saving
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Return a dictionary with the same structure as input data
    return {'images': val_images, 'labels': val_labels}, {'images': train_images, 'labels': train_labels}

