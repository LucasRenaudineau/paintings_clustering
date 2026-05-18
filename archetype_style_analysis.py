import numpy as np
import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from archetypes import AA
from preprocessing import *
from glob import glob
import gc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from archetype_visualisation import *

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

paths = [path for path in glob("ArtemisArt/**/*.jpg") if path.split("/")[1][0] == "a"]
print(f"Il y a {len(paths)} images dans le dataset")


class ArchetypeGenerator:
    def __init__(self, nb_archetypes):
        self.k = nb_archetypes
        self.archetype = AA(nb_archetypes)

    def find_archetypes(self, data_path):
        transformed_features = []
        for i, path in enumerate(data_path):
            with torch.no_grad():
                img = imread_safe(path)
                print(f"Image n° {i}, taking features from {path}")
                """This code is taken from features_extractor.py"""
                h, w = img.shape[:2]
                if w != 1800 and h != 1800:
                    continue
                preprocess_img = preprocessing_image(img)
                tensor_img = preprocess_img.to(device)
                feature_maps = feature_extractor(tensor_img).values()
                # print(feature_maps)
                # print(f"shape : {len(feature_maps)} x {feature_maps[0].shape}")
                x_raw_list = []
                for torch_f_map in feature_maps:
                    f_map = torch_f_map.detach().cpu().float().numpy()[0]
                    f_map = f_map.reshape(
                        f_map.shape[0], f_map.shape[1] * f_map.shape[2]
                    )
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
                del img, preprocess_img, tensor_img, feature_maps
                if i % 10 == 0:
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                transformed_features.append(np.concatenate(x_raw_list).flatten())
        # U, S, VH = np.linalg.svd(np.stack(transformed_features), full_matrices=False)
        X_raw = np.stack(transformed_features)
        pca = PCA(min(4096, X_raw.shape[0] - 1))
        X = pca.fit_transform(X_raw)
        print(f"shape of X : {X.shape}")
        self.X = X
        self.pca = pca
        A = self.archetype.fit_transform(X)
        Z = self.archetype.archetypes_
        B = self.archetype.B_
        self.A = A
        self.B = B
        self.Z = Z
        return A, B, Z, pca

    def generate_archetypes(self):
        pass

    pass


a = ArchetypeGenerator(4)
A, B, Z, pca_mod = a.find_archetypes(paths)

print(f"Voila la forme de A : {A}\n")
print(f"Voila la forme de B : {B}\n")
print(f"Voila la forme de Z : {Z}\n")

# On choisit l'Archétype 0
Z_arch_0 = Z[0]

# Et on lance !
image_synthetisee = synthetiser_archetype(
    Z_archetype=Z_arch_0,
    pca_model=pca_mod,
    extractor=feature_extractor,
    device=device,
)
plt.imshow(image_synthetisee)
plt.axis("off")
plt.title("L'essence de l'Archétype 0")
plt.savefig("archetype_0.png", bbox_inches="tight", dpi=300)
print("Image sauvegardée sous le nom 'archetype_0.png' !")

plt.close()
