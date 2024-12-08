import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import numpy as np
import random
import keras_cv as kcv
import cv2
import scipy.ndimage as ndimage
import skimage.transform as transform
from IPython.display import display, clear_output

def augment(image):
    """
    Given an image, using `augment(image)` will return a new image
    that is the augmentation of the one in input.
    Function return a uint8 image.
    """

    def random_blur(image, max_radius=1.0):
        image_np = image.numpy()
        radius = tf.random.uniform([], 0, max_radius, dtype=tf.float32).numpy()
        blurred_image_np = cv2.GaussianBlur(image_np, (5, 5), radius)
        return tf.convert_to_tensor(blurred_image_np)

    # List of augmentation functions that are applied to the image
    augmentation_types = [
        random_blur,
        tfkl.RandomFlip(),
        tfkl.RandomRotation(1),
        tfkl.RandomBrightness(0.2, value_range=(0,1)),
        kcv.layers.RandomHue(value_range=(0, 1), factor=1.0),
        kcv.layers.AutoContrast(value_range=(0, 1)),
        kcv.layers.GridMask(),
        kcv.layers.RandomColorDegeneration(factor=1.0),
        kcv.layers.RandomCutout(width_factor=0.5, height_factor=0.5),
        kcv.layers.RandomSaturation(factor=(0.2, 0.8)),
        kcv.layers.RandomShear(x_factor=0.5, y_factor=0.5),
        # kcv.layers.AugMix(value_range=(0, 1)),
        # kcv.layers.RandAugment(value_range=(0, 1)),
    ]

    img_tf32 = tf.image.convert_image_dtype(image, dtype=tf.float32)
    subset = random.sample(augmentation_types, random.randint(1, len(augmentation_types)))
    for type in subset:
        img_tf32 = type(img_tf32)
    img_tf8 = tf.image.convert_image_dtype(img_tf32, dtype=tf.uint8)
    return np.array(img_tf8.numpy(), dtype=np.uint8)

def augment_set(data, surplus=1, top=-1):
    """
    Given a dataset, using `augment_set(data)` will return a
    new balanced dataset which
    - does not contains the data of the input dataset,
    - has all the classes balanced,
    - all classes contain only augmented images,
    - images coming from the same origin are consequent.

    The argument `surplus` is multiplied to the number of images
    of the biggest class for determining how many images will be
    augmented, for each class, in the end (e.g. `surplus=1.5`).

    The argument `top` directly sets the number of images
    that are to be augmented for each class (e.g. `top=2000`).
    If `top` is set, `surplus` is ignored.
    """
    # Retrieve images and labels from dataset
    images = data['images']
    labels = data['labels']

    # Count how many images are there for each label
    classes = len(np.unique(labels))
    counts = np.bincount(labels.flatten(), minlength=classes)

    # Set the desired number of images for each label
    if top <= 0:
        roof = int(counts.max() * surplus)
    else:
        roof = top

    # Get indices for each label (indices[label][index])
    indices = [np.where(labels.flatten() == i)[0] for i in range(classes)]

    # Initialize lists for augmented dataset
    new_images = []
    new_labels = []

    # Augment images to balance the dataset
    for label, count in enumerate(counts):
        for i, idx in enumerate(indices[label]):
            image = images[idx]

            # Augment image
            for _ in range(roof // count + (1 if i < roof % count else 0)):
                new_images.append(augment(image))
                new_labels.append(label)

            # Print progress
            clear_output(wait=True)
            print(f"Class {label+1}/{classes} - Augmenting {i+1}/{count}")

    # Convert to numpy arrays for saving
    new_images = np.array(new_images)
    new_labels = np.array(new_labels, dtype=np.uint8).reshape(-1, 1)

    # Return a dictionary with the same structure as input data
    return {'images': new_images, 'labels': new_labels}

def masked_augment(image, mask):
    """
    Given an image, using `masked_augment(image, mask)` will return a new pair
    image/mask that are the augmentation of the ones in input.
    """

    def random_flip(image, mask, p_horizontal=0.5, p_vertical=0.5):
        if np.random.random() < p_horizontal:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        if np.random.random() < p_vertical:
            image = np.flipud(image)
            mask = np.flipud(mask)
        return image, mask

    def random_brightness_contrast(image, brightness_range=(-50, 50), contrast_range=(0.5, 1.5)):
        image = image.astype(np.float32)
        brightness = np.random.uniform(brightness_range[0], brightness_range[1])
        image_bright = image + brightness
        contrast = np.random.uniform(contrast_range[0], contrast_range[1])
        image_contrast = (image_bright - np.mean(image_bright)) * contrast + np.mean(image_bright)
        return np.clip(image_contrast, 0, 255).astype(np.uint8)

    def random_gaussian_noise(image, mean=0, std_range=(1, 5)):
        image = image.astype(np.float32)
        std = np.random.uniform(std_range[0], std_range[1])
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def random_elastic_deformation(image, mask, alpha=50, sigma=5):
        image_float = image.astype(np.float32)
        mask_float = mask.astype(np.float32)
        shape = image.shape
        dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        deformed_x = x + dx
        deformed_y = y + dy
        deformed_image = ndimage.map_coordinates(image_float, [deformed_y, deformed_x], order=1, mode='nearest')
        deformed_mask = ndimage.map_coordinates(mask_float, [deformed_y, deformed_x], order=1, mode='nearest')
        return np.clip(deformed_image, 0, 255).astype(np.uint8), np.clip(deformed_mask, 0, 255).astype(np.uint8)

    aug_image, aug_mask = image.copy(), mask.copy()

    aug_image, aug_mask = random_flip(aug_image, aug_mask)
    aug_image = random_brightness_contrast(aug_image)
    aug_image = random_gaussian_noise(aug_image)
    aug_image, aug_mask = random_elastic_deformation(aug_image, aug_mask)

    return aug_image, aug_mask

def augment_masked_set(data, luck_div = 1.3):
    """
    Given a dataset, using `augment_masked_set(data)` will return a
    new dataset which
    - contains also the data of the input dataset,
    - helps with dataset balance by augmenting scarse images more often,
    - images coming from the same origin are consequent.

    The parameter `luck_div` represents the divider value that is
    multiplied to the augmentation probability on each round, after the
    first, when augmenting the same image again (e.g `luck_div=1.3`).

    """
    # Retrieve images and labels from dataset
    images = data['images']
    labels = data['labels']

    # Find count of recurrencies of a class in each image
    v = []
    for label in labels:
        v = np.append(v, np.unique(label))
    v = v.astype(int)
    counts = np.bincount(v)

    # Build a new dataset with both original and augmented images
    new_images = []
    new_labels = []
    tot = len(labels)
    for i, (image, label) in enumerate(zip(images, labels)):

        # Print progress
        clear_output(wait=True)
        print(f"Augmenting image {i+1}/{tot}")

        # Append original image
        new_images.append(image)
        new_labels.append(label)

        # Find class count in dataset of higher class in label
        higher_class = np.max(label)
        count = counts[higher_class]

        # Calculate probability of making an augmentation
        prob = 1 - count / tot
        sampled = np.random.random()
        while sampled <= prob:
            aug_image, aug_label = masked_augment(image, label)

            # Append augmented image
            new_images.append(aug_image)
            new_labels.append(aug_label)

            # Another round parameters
            sampled = np.random.random()
            prob = prob / luck_div

    # Convert to numpy arrays for saving
    new_images = np.array(new_images)
    new_labels = np.array(new_labels)

    # Return a dictionary with the same structure as input data
    return {'images': new_images, 'labels': new_labels}

