import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
import re
import matplotlib.pyplot as plt


def retina_preprocessing(image_path):
    """
    Preprocesses a retina image file.

    Args:
        image_path: a string, the path to the image file

    Returns:
        a tensor of shape (1, 224, 224, 3), the preprocessed image
    """
    image = plt.imread(image_path)
    # If the image is 2D (grayscale), add a channel dimension
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.grayscale_to_rgb(image)

    image = tf.image.resize(image, [224, 224])
    image = preprocess_input(image)
    return tf.expand_dims(image, axis=0)

def pneumonia_preprocessing(image_path):
    
    """
    Preprocesses a chest X-ray image file.

    Args:
        image_path: a string, the path to the image file

    Returns:
        a tensor of shape (1, 224, 224, 3), the preprocessed image
    """
    
    image = plt.imread(image_path)
    
    # If the image is 2D (grayscale), add a channel dimension
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.grayscale_to_rgb(image)
    
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)


def kidney_brain_preprocessing(image_path):
    """
    Preprocesses a kidney image file.

    Args:
        image_path: str, the path to the kidney image file

    Returns:
        tensor: a tensor of shape (1, 224, 224, 1), the preprocessed kidney image
    """
    image = plt.imread(image_path)
    
    # If the image is 2D (grayscale), add a channel dimension
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)

    image = tf.image.resize(image, [224, 224])

    # if not grayscale convert it
    if image.shape[-1] != 1:
        image = tf.image.rgb_to_grayscale(image)
    return tf.expand_dims(image, axis=0)



def general_preprocessing(image_path):

    image = plt.imread(image_path)
    # If the image is 2D (grayscale), add a channel dimension
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.grayscale_to_rgb(image)

    image = tf.image.resize(image, [224, 224])
    image = inception_preprocess_input(image)
    return tf.expand_dims(image, axis=0)


def markdown_to_text(markdown):
    # Remove headers
    markdown = re.sub(r'^#{1,6}\s*', '', markdown, flags=re.MULTILINE)
    # Remove bold
    markdown = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'\1', markdown)
    # Remove italics
    markdown = re.sub(r'\*(.*?)\*|_(.*?)_', r'\1', markdown)
    # Remove links
    markdown = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', markdown)
    # Remove inline code
    markdown = re.sub(r'`([^`]+)`', r'\1', markdown)
    # Remove images
    markdown = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', markdown)
    # Remove lists
    markdown = re.sub(r'^[*-]\s+', '', markdown, flags=re.MULTILINE)
    # Remove blockquotes
    markdown = re.sub(r'^>\s*', '', markdown, flags=re.MULTILINE)
    # Remove horizontal rules
    markdown = re.sub(r'^-{3,}|^\*{3,}', '', markdown, flags=re.MULTILINE)
    # Remove strikethrough
    markdown = re.sub(r'~~(.*?)~~', r'\1', markdown)

    return markdown.strip()