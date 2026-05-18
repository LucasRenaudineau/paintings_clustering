import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
from archetypes import AA
from preprocessing import *
from glob import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# We use vgg19_bn instead of vgg19 since it is better
model = models.vgg19_bn(weights=models.VGG19_BN_Weights)
model = model.to(device)
model.eval()

return_nodes = {
    "features.2": "layer1",
    "features.5": "layer2",
    "features.9": "layer3",
    "features.12": "layer4",
    "features.16": "layer5",
}

feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

paths = glob("ArtemisArt/delacroix*/*.jpg")
print(f"Il y a {len(paths)} images dans le dataset")
feature_maps_list = []
for path in paths:
    img = imread_safe(path)
    print(f"Taking features from {path}")
    """This code is taken from features_extractor.py"""
    with torch.no_grad():
        preprocess_img = preprocessing_image(img)
        tensor_img = preprocess_img.to(device)
        feature_maps = feature_extractor(tensor_img).values()
        # print(feature_maps)
        # print(f"shape : {len(feature_maps)} x {feature_maps[0].shape}")
        feature_maps_list.append(feature_maps)


class ArchetypeGenerator:
    def __init__(self, nb_archetypes):
        self.k = nb_archetypes
        self.archetype = AA(nb_archetypes)

    def find_archetypes(self, feature_maps_list):
        transformed_features = []
        for i, feature_maps in enumerate(feature_maps_list):
            print(f"Transforming data n°{i}")
            x_raw_list = []
            for torch_f_map in feature_maps:
                f_map = torch_f_map.detach().cpu().float().numpy()[0]
                f_map = f_map.reshape(f_map.shape[0], f_map.shape[1] * f_map.shape[2])
                p, m = f_map.shape[0], f_map.shape[1]
                # print(f"THE VALUE OF p IS {p} !!!!!!!")
                mu = np.mean(f_map, axis=1).reshape(-1, 1)
                # print(f"f_map shape: {f_map.shape}")
                # print(f"mu shape: {mu.shape}")
                sigma = (f_map - mu) @ (f_map - mu).T / m
                mu = mu / (p * (p - 1))
                sigma = sigma / (p * (p - 1))
                sigma_flat = sigma.flatten()
                # print(f"The shape of sigma_flat is : {sigma_flat.shape}")
                x_raw = np.concatenate([mu, sigma_flat.reshape(-1, 1)])
                # print(f"shape of x_raw : {x_raw.shape}")
                x_raw_list.append(x_raw)
            # print(len(x_raw_list))
            # print(f"Shape of the x_raw_list : {np.concatenate(x_raw_list).shape}")
            U, S, VH = np.linalg.svd(np.concatenate(x_raw_list), full_matrices=False)
            # print("Shape for SVD:")
            # print(U.shape, S.shape, VH.shape)
            x = U[:4096, :].reshape(1, -1).flatten()
            # print(f"shape of x : {x.shape}")
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


a = ArchetypeGenerator(4)
A, B, Z = a.find_archetypes(feature_maps_list)

print(f"Voila la forme de A : {A}\n")
print(f"Voila la forme de B : {B}\n")
print(f"Voila la forme de Z : {Z}\n")
