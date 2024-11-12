import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import cv2
from IPython.display import display, clear_output

def random_stretch(image, max_scale=1.5):
    scale = tf.random.uniform([], 1.0, max_scale)
    if tf.random.uniform([]) > 0.5:
        new_width, new_height = tf.cast(96 * scale, tf.int32), 96
    else:
        new_width, new_height = 96, tf.cast(96 * scale, tf.int32)
    resized_image = tf.image.resize(image, [new_height, new_width])
    return tf.image.resize_with_crop_or_pad(resized_image, 96, 96)

def random_hue(image, max_delta=0.5):
    delta = tf.random.uniform([], minval=-max_delta, maxval=max_delta)
    return tf.image.adjust_hue(image, delta)

def random_blur(image, max_radius=1.0):
    image_np = image.numpy()
    radius = tf.random.uniform([], 0, max_radius, dtype=tf.float32).numpy()
    blurred_image_np = cv2.GaussianBlur(image_np, (5, 5), radius)
    return tf.convert_to_tensor(blurred_image_np)

def random_saturation(image, max_factor=0.5):
    factor = tf.random.uniform([], minval=1-max_factor, maxval=1+max_factor)
    return tf.image.adjust_saturation(image, factor)

# List of augmentation functions that are applied to the image
augmentation_types = [
    tfkl.RandomFlip(),
    tfkl.RandomRotation(1),
    tfkl.RandomBrightness(0.2, value_range=(0,1)),
    tfkl.RandomContrast(0.75),
    random_stretch,
    random_hue,
    random_blur,
    random_saturation
]

def augment(image):
    """
    Given an image, using `augment(image)` will return a new image 
    that is the augmentation of the one in input.
    """
    img_tf32 = tf.image.convert_image_dtype(image, dtype=tf.float32)
    for type in augmentation_types:
        img_tf32 = type(img_tf32)
    img_tf8 = tf.image.convert_image_dtype(img_tf32, dtype=tf.uint8)
    return np.array(img_tf8.numpy(), dtype=np.uint8)

def augment_set(data, surplus=0.1):
    """
    Given a dataset, using `augment_set(data)` will return a 
    new balanced dataset which
    - contains all the data present in the input dataset,
    - has all the classes balanced,
    - all classes contain some augmented images,
    - each image is followed by its augmented versions.

    The argument `surplus` can be used to set the percentage of the 
    biggest class that has also to be augmented; other classes will 
    have the same number of elements as the biggest class, since it 
    has to be balanced.
    """
    # Retrieve images and labels from dataset
    images = data['images']
    labels = data['labels']

    # Count how many images are there for each label
    classes = len(np.unique(labels))
    counts = np.bincount(labels.flatten(), minlength=classes)

    # Set the desired number of images for each label
    roof = int(counts.max() * (1 + surplus))

    # Get indices for each label (indices[label][index])
    indices = [np.where(labels.flatten() == i)[0] for i in range(classes)]

    # Initialize lists for augmented dataset
    new_images = []
    new_labels = []

    # Augment images to balance the dataset
    for label, count in enumerate(counts):
        for i, idx in enumerate(indices[label]):

            # Append original image
            image = images[idx]
            new_images.append(image)
            new_labels.append(label)

            # Augment image
            for n in range((roof-i) // count):
                augmented_image = augment(image)
                new_images.append(augmented_image)
                new_labels.append(label)

                # Print progress
                clear_output(wait=True)
                print(f"Class {label+1}/{classes} - Original: {count} - Target: {roof} - Augmenting {i}/{min(count, roof-count)}")

    # Convert to numpy arrays for saving
    new_images = np.array(new_images)
    new_labels = np.array(new_labels, dtype=np.uint8).reshape(-1, 1)

    # Return a dictionary with the same structure as input data
    return {'images': new_images, 'labels': new_labels}

def split_set(augmented_data, val=0.2, shuffle=True, starting=-1):
    """
    Given the augmented dataset, using `split_set(augmented_data)`
    will return two datasets, one for training and one with validation.
    Generally, if a set contains an augmented image, it contains 
    also its original version, and viceversa.
    Validation set is returned first, training set as second.

    The argument `val` can be used to set the percentage of the 
    dataset to reserve for validation (e.g. `val=0.2`).

    The argument `shuffle` can be used to shuffle the two returned 
    sets, as otherwise the classes are found in blocks one after 
    the other (e.g. `shuffle=True`).

    The argument `starting` can be used to set the starting point 
    (represented as a percentage) from which to start picking 
    images for the validation set (e.g. `starting=0.6`).  
    """
    # Retrieve images and labels from augmented dataset
    images = augmented_data['images']
    labels = augmented_data['labels']

    # Select starting point from which to pick images
    max_count = np.count_nonzero(labels == 0)
    if (starting == -1 ):
        start = random.randint(0, max_count - 1)
    else:
        start = int(max_count * starting)

    # Define sizes
    val_size = int(max_count * val)
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

def plot_image(*images):
    """
    Images are plotted horizontally.
    One or many images can be passed as arguments.
    Usage:
    - `plot_image(image)` 
    - `plot_image(image, augmented_image)`
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
    for ax, image in zip(axes, images):
        ax.imshow(image)
        ax.axis('off')
    plt.show()

def plot_distribution(*datas):
    """
    Graphs are plotted horizontally.
    One or many label sets can be passed as arguments.
    Usage:
    - `plot_distribution(data['labels'])` 
    - `plot_distribution(data_labels, augmented_labels)`
    """
    n = len(datas)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
    if n == 1:
        axes = [axes]
    for ax, data in zip(axes, datas):
        classes = len(np.unique(data))
        counts = np.bincount(data.flatten(), minlength=classes)
        bars = ax.bar(range(classes), counts, tick_label=[i for i in range(classes)])
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, str(int(yval)), ha='center', va='bottom')
        ax.set_ylabel("Count")
        ax.set_xlabel("Label")
        ax.set_xticklabels([i for i in range(classes)])
    plt.tight_layout()
    plt.show()

def print_example_images(images, labels, row=8):
    """
    Initial images and final images of each label are
    plotted horizontally.
    Usage:
    - `print_example_images(images, labels)`
    - `print_example_images(images, labels, row=8)`
    """
    classes = len(np.unique(labels))
    for i in range(classes):
        print(f"Class: {i+1}/{classes}")
        indices = np.where(labels == i)[0]
        plot_image(*images[indices[:row]])
        plot_image(*images[indices[-row:]])










