"""Same file as features extractor but using PyTorch."""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor

from preprocessing import *

MODEL_PATH = "./models/resnet50.pth"
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
        weights_dict = torch.load(MODEL_PATH)
        model = models.resnet50(weights=weights_dict)
        model.fc = nn.Identity()
        print(f"Loaded ResNet50 from {MODEL_PATH}")
    except (OSError, IOError, ValueError):
        print(f"No saved model found at {MODEL_PATH}. Downloading ResNet50 with ImageNet weights.")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity() # include_top=False
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    model=model.to(device)
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
        conv1_relu                -> (1, 112, 112, 64)
        conv2_block3_out          -> (1, 56, 56, 256)
        conv3_block4_out          -> (1, 28, 28, 512)
        conv4_block6_out          -> (1, 14, 14, 1024)
        conv5_block3_out          -> (1, 7, 7, 2048)
    """
    
    tensor_preprocess_image = preprocessing_image(image)
    tensor_preprocess_image=tensor_preprocess_image.to(device)

    torch.no_grad()
    activations = feature_model(tensor_preprocess_image)
    numpy_activations = {}
    for layer_name, tensor in activations.items():
        # Go back to (1,H,W,C)
        numpy_activations[layer_name] = tensor.cpu().detach().numpy().transpose(0, 2, 3, 1)
 
    return numpy_activations

if __name__ == "__main__":
    _, feature_model = load_model()
    test_path = "ArtemisArt/afro - afro-basaldella_1912/afro_1.jpg"
    img = imread_safe(test_path)
    features = extract_features(img, feature_model)
    print("Features shapes :")
    for layer, activation in features.items():
        print(f"{layer:25s} -> {str(activation.shape)}")




