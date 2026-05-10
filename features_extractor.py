import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
 
from resize import *
 
# Constants
 
MODEL_PATH = "./models/resnet50.keras"
INPUT_SHAPE = (224, 224) # ResNet50 input size
 
# Intermediate layers whose activations we want to extract.
FEATURE_LAYER_NAMES = [
    "conv1_relu", # Stage 1 — low-level edges & textures
    "conv2_block3_out", # Stage 2 — simple shapes
    "conv3_block4_out", # Stage 3 — mid-level patterns
    "conv4_block6_out", # Stage 4 — high-level semantics
    "conv5_block3_out", # Stage 5 — near-final representations
]
 
# Load model
 
def load_model() -> tuple[Model, Model]:
    """
    Load ResNet50 from ./models/resnet50 if it exists, otherwise download the
    ImageNet weights and save them there for future runs.
 
    Returns:
        base_model: the full ResNet50 Keras model.
        feature_model: a multi-output Model that returns the activations of
                       every layer listed in FEATURE_LAYER_NAMES.
    """
    try:
        base_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Loaded ResNet50 from {MODEL_PATH}")
    except (OSError, IOError, ValueError):
        print(f"No saved model found at {MODEL_PATH}. Downloading ResNet50 with ImageNet weights.")
        base_model = ResNet50(weights="imagenet", include_top=False) # We don't need the classifying layers
        base_model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
 
    outputs = [base_model.get_layer(name).output for name in FEATURE_LAYER_NAMES]
    feature_model = Model(inputs=base_model.input, outputs=outputs)
 
    return base_model, feature_model

# Image preprocessing
 
def resize_image(image: np.ndarray) -> np.ndarray:
    """
    Pad then resize a BGR image to the 224x224 input size expected by ResNet50.
 
    Args:
        image: NumPy array of shape (H, W, 3), BGR uint8.
               The longest dimension must be 1800px.
 
    Returns:
        NumPy array of shape (1, 224, 224, 3), float32, ready for inference.
    """
    squared = pad_to_square(image)
    rgb = squared[:, :, ::-1]
    resized = tf.image.resize(rgb, INPUT_SHAPE).numpy().astype(np.float32) # resized to 224*224
 
    batched = np.expand_dims(resized, axis=0) # adding batch dimension : (1, 224, 224, 3)
    return preprocess_input(batched) # applying resnet50 preprocessing

# Features extraction

def extract_features(image: np.ndarray, feature_model: Model) -> dict[str, np.ndarray]:
    """
    Run a single image through the multi-output feature model and return the
    activation maps of every intermediate layer.
 
    Args:
        image: Raw image as a NumPy array (H, W, 3), BGR uint8.
        feature_model: The multi-output Keras Model returned by load_model().
 
    Returns:
        Dictionary mapping each layer name to its activation NumPy array.
        Shapes (with default FEATURE_LAYER_NAMES):
            "conv1_relu" -> (1, 112, 112, 64)
            "conv2_block3_out" -> (1, 56, 56, 256)
            "conv3_block4_out" -> (1, 28, 28, 512)
            "conv4_block6_out" -> (1, 14, 14, 1024)
            "conv5_block3_out" -> (1, 7, 7, 2048)
            "avg_pool" -> (1, 2048)
    """
    preprocessed = resize_image(image)
    activations = feature_model.predict(preprocessed, verbose=0)
 
    return {layer_name: activation for layer_name, activation in zip(FEATURE_LAYER_NAMES, activations)}

if __name__ == "__main__":
    _, feature_model = load_model()
    test_path = "ArtemisArt/afro - afro-basaldella_1912/afro_1.jpg"
    img = imread_safe(test_path)
    features = extract_features(img, feature_model)
    print("Features shapes :")
    for layer, activation in features.items():
        print(f"{layer:25s} -> {str(activation.shape)}")
