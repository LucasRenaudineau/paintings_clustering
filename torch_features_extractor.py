"""Same file as features extractor but using PyTorch."""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor

from resize import *

MODEL_PATH = "./models/resnet50.keras"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Used ai to know the corresponding names for pytorch
FEATURE_LAYER_NAMES = {
    "relu": "conv1_relu",             # Stage 1 — low-level edges & textures
    "layer1.2": "conv2_block3_out",   # Stage 2 — simple shapes (fin du 3e bloc)
    "layer2.3": "conv3_block4_out",   # Stage 3 — mid-level patterns (fin du 4e bloc)
    "layer3.5": "conv4_block6_out",   # Stage 4 — high-level semantics (fin du 6e bloc)
    "layer4.2": "conv5_block3_out",   # Stage 5 — near-final representations (fin du 3e bloc)
}

def load_model():
    """
    Load ResNet50 from ./models/resnet50 if it exists, otherwise download the
    ImageNet weights and save them there for future runs.
 
    Returns:
        base_model: the full ResNet50 Keras model.
        feature_model: a multi-output Model that returns the activations of
                       every layer listed in FEATURE_LAYER_NAMES.
    """
    try:
        model = torch.load(MODEL_PATH)
        print(f"Loaded ResNet50 from {MODEL_PATH}")
    except (OSError, IOError, ValueError):
        print(f"No saved model found at {MODEL_PATH}. Downloading ResNet50 with ImageNet weights.")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity() # include_top=False
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    model.eval()
    feature_model = create_feature_extractor(model, return_nodes=FEATURE_LAYER_NAMES)
    feature_model.eval()

    return model, feature_model

def extract_features(image, feature_model):
    """
    Run a single image through the multi-output feature model and return the
    activation maps of every intermediate layer.
 
    Args:
        image: Raw image as a NumPy array (H, W, 3), BGR uint8.
        feature_model: The multi-output Keras Model returned by load_model().
 
    Returns:
        Dictionary mapping each layer name to its activation NumPy array.
    """
    # Used AI in this function to help me convert it for pytorch
    preprocessed = resize_image(image)
    image_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

    # In pytorch, it's channel first.
    tensor_img = torch.from_numpy(image_rgb).permute(2, 0, 1).float()

    # Normalize
    tensor_img = tensor_img / 255.0
    tensor_img = F.normalize(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_img = tensor_img.unsqueeze(0)
    tensor_img=tensor_img.to(device)

    torch.no_grad()
    activations = feature_model(tensor_img)
    numpy_activations = {}
    for layer_name, tensor in activations.items():
        # Go back to (1,H,W,C)
        numpy_activations[layer_name] = tensor.cpu().numpy().transpose(0, 2, 3, 1)
 
    return numpy_activations

if __name__ == "__main__":
    _, feature_model = load_model()
    test_path = "ArtemisArt/afro - afro-basaldella_1912/afro_1.jpg"
    img = imread_safe(test_path)
    features = extract_features(img, feature_model)
    print("Features shapes :")
    for layer, activation in features.items():
        print(f"{layer:25s} -> {str(activation.shape)}")




