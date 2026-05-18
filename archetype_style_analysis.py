import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
from archetypes import AA
from resize import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# We use vgg19_bn instead of vgg19 since it is better
model = models.vgg19_bn(weights=models.VGG19_BN_Weights)
model = model.to(device)

FEATURE_LAYER_NAMES = []
FEATURE_LAYER_MODULES = []
# Extracting layer names, it has the format : name.number
# We are only interested in the features part so we neglict the classifier part.
for name, module in model.named_modules():
    if "Conv2d" in str(module) or "Linear" in str(module):
        if len(name) > 0 and "." in name and "classifier" not in name:
            # print(name, ":", module)
            FEATURE_LAYER_NAMES.append(name)
            FEATURE_LAYER_MODULES.append(module)
print(f"layers :\n {FEATURE_LAYER_NAMES}")
print(f"Modules :\n{FEATURE_LAYER_MODULES}")


def extract_feature_maps(input, feature_modules):
    curr = input
    feature_maps = []
    for i in range(5):  # We extract only the five first layers.
        curr = feature_modules[i](curr)
        feature_maps.append(curr)
    return feature_maps


test_path = "ArtemisArt/afro - afro-basaldella_1912/afro_1.jpg"
img = imread_safe(test_path)

"""This code is taken from features_extractor.py"""
preprocessed = resize_image(img)
image_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

# In pytorch, it's channel first.
tensor_img = torch.from_numpy(image_rgb).permute(2, 0, 1).float()

# Normalize
tensor_img = tensor_img / 255.0
tensor_img = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)(tensor_img)
tensor_img = tensor_img.unsqueeze(0)
tensor_img = tensor_img.to(device)
torch.no_grad()
feature_maps = extract_feature_maps(tensor_img, FEATURE_LAYER_MODULES)
print(feature_maps)
print(f"shape : {len(feature_maps)} x {feature_maps[0].shape}")


class ArchetypeGenerator:
    def __init__(self, nb_archetypes):
        self.k = nb_archetypes
        self.archetype = AA(nb_archetypes)

    def find_archetypes(self, feature_maps):
        transformed_features = []
        for f_map in feature_maps:
            p, m = f_map.shape[0], f_map.shape[1]
            mu = np.mean(f_map, axis=1)
            sigma = np.sum((f_map - mu) @ (f_map - mu).T, axis=1) / m
            mu = mu / (p * (p - 1))
            sigma = sigma / (p * (p - 1))
            sigma_flat = sigma.flatten()
            x_raw = np.concatenate(mu, sigma_flat).reshape(-1, 1)
            U, S, VH = np.linalg.svd(x_raw, full_matrices=False)
            x = U[:, :4096].reshape(1, -1).flatten()
            transformed_features.append(x)
        X = np.array(transformed_features)
        self.X = X
        A = self.archetype.fit_transform(X)
        Z = self.archetype.archetypes_
        B = self.archetype.B_
        self.A = A
        self.B = B
        self.Z = Z
        return A, B, Z

    def generate_archetypes(self):
        pass

    pass


a = ArchetypeGenerator(8)
print(a.find_archetypes(feature_maps))
