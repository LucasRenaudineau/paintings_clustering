import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor

from preprocessing import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# We use vgg19_bn instead of vgg19 since it is better 
model = models.vgg19_bn(weights=models.VGG19_BN_Weights)
model=model.to(device)

FEATURE_LAYER_NAMES = []
FEATURE_LAYER_MODULES = []
# Extracting layer names, it has the format : name.number
# We are only interested in the features part so we neglict the classifier part.
for name, module in model.named_modules():
    if "Conv2d" in str(module) or "Linear" in str(module):
        if len(name)>0 and "." in name and "classifier" not in name:
            # print(name, ":", module)
            FEATURE_LAYER_NAMES.append(name)
            FEATURE_LAYER_MODULES.append(module)
print(f"layers :\n {FEATURE_LAYER_NAMES}")
print(f"Modules :\n{FEATURE_LAYER_MODULES}")
def extract_feature_maps(input, feature_modules):
    curr = input
    feature_maps=[]
    for i in range(5): # We extract only the five first layers.
        curr=feature_modules[i](curr)
        feature_maps.append(curr)
    return feature_maps
test_path = "ArtemisArt/afro - afro-basaldella_1912/afro_1.jpg"
img = imread_safe(test_path)

"""This code is taken from features_extractor.py"""
preprocess_img = preprocessing_image(img)
tensor_img=preprocess_img.to(device)
torch.no_grad()
feature_maps=extract_feature_maps(tensor_img, FEATURE_LAYER_MODULES)
print(feature_maps)
print(f"shape : {len(feature_maps)} x {feature_maps[0].shape}")

class ArchetypeGenerator:
    def __init__(self):
        pass
    def generate_archetypes(self):
        pass
    pass