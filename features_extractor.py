import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
 
from resize import pad_to_square
 
# Constants
 
MODEL_PATH = "./models/resnet50.keras"
INPUT_SHAPE = (224, 224)  # ResNet50 input size
 
# Intermediate layers whose activations we want to extract.
FEATURE_LAYER_NAMES = [
    "conv1_relu",          # Stage 1 — low-level edges & textures
    "conv2_block3_out",    # Stage 2 — simple shapes
    "conv3_block4_out",    # Stage 3 — mid-level patterns
    "conv4_block6_out",    # Stage 4 — high-level semantics
    "conv5_block3_out",    # Stage 5 — near-final representations
    "avg_pool",            # Global average pool — compact descriptor
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

if __name__ == "__main__":
    base_model, feature_model = load_model()
