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
    "relu": "conv1_relu",
    "layer1.2": "conv2_block3_out",
    "layer2.3": "conv3_block4_out",
    "layer3.5": "conv4_block6_out",
    "layer4.2": "conv5_block3_out",
}

def load_model():
    """
    Load ResNet50 from ./models/resnet50.pth if it exists, otherwise download the
    ImageNet weights and save them there for future runs.
    """
    try:
        model = models.resnet50(weights=None)
        model.fc = nn.Identity()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
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

def _run_feature_model(tensor, feature_model):
    """Run a preprocessed (1,3,224,224) tensor through the multi-output model
    and return the activations as a list of (1, H, W, C) NumPy arrays."""
    tensor = tensor.to(device)
    with torch.no_grad():
        activations = feature_model(tensor)
    numpy_activations = []
    for layer_name, t in activations.items():
        # Go back to (1,H,W,C)
        numpy_activations.append(t.cpu().detach().numpy().transpose(0, 2, 3, 1))
    return numpy_activations


def extract_features(image, feature_model):
    """
    Returns:
        Array mapping each layer name in order to its activation NumPy array.
            0  -> (1, 112, 112, 64)      (conv1_relu)
            1  -> (1, 56, 56, 256)       (conv2_block3_out)
            2  -> (1, 28, 28, 512)       (conv3_block4_out)
            3  -> (1, 14, 14, 1024)      (conv4_block6_out)
            4  -> (1, 7, 7, 2048)        (conv5_block3_out)
    """
    return _run_feature_model(preprocessing_image(image), feature_model)


def extract_crop_features(image, feature_model):
    crop = find_most_uniform_crop(image)
    crop_gray = to_grayscale_3ch(crop) # just comment this line to keep colors
    return _run_feature_model(preprocessing_image(crop_gray, pad=False), feature_model)


def extract_all_features(image, feature_model):
    """
        0..4  -> full-image activations    (conv1_relu .. conv5_block3_out), colour
        5..9  -> uniform-crop activations  (same layers), grayscale texture
    """
    return extract_features(image, feature_model) + extract_crop_features(image, feature_model)


if __name__ == "__main__":
    _, feature_model = load_model()
    test_path = "ArtemisArt/afro - afro-basaldella_1912/afro_1.jpg"
    img = imread_safe(test_path)
    features = extract_all_features(img, feature_model)
    print("Features shapes (0-4 full image, 5-9 uniform crop):")
    for activation in features:
        print(f"{str(activation.shape)}")
