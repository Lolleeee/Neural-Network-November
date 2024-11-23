# %% [code]
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import numpy as np
import random
import keras_cv as kcv
import cv2
from IPython.display import display, clear_output

def augment(image):
    """
    Given an image, using `augment(image)` will return a new image 
    that is the augmentation of the one in input.
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
        kcv.layers.AugMix(value_range=(0, 1)),
        kcv.layers.GridMask(),
        kcv.layers.RandomColorDegeneration(factor=1.0),
        kcv.layers.RandomCutout(width_factor=0.5, height_factor=0.5),
        kcv.layers.RandomSaturation(factor=(0.2, 0.8)),
        kcv.layers.RandomShear(x_factor=0.5, y_factor=0.5),
        # kcv.layers.RandAugment(value_range=(0, 1)), # maybe this is too much
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
