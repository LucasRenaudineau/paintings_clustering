import os
import numpy as np
import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from archetypes import AA
from preprocessing import *
from glob import glob
import gc
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from archetype_visualisation import *
from tqdm import tqdm


class ArchetypeGenerator:
    def __init__(self, nb_archetypes, data_path, device, feature_extractor):
        self.k = nb_archetypes
        self.archetype = AA(nb_archetypes)
        self.data_path = np.array(data_path)
        self.feature_extractor = feature_extractor
        self.device = device

    def transform(self, img):
        with torch.no_grad():
            preprocess_img = preprocessing_image(img)
            tensor_img = preprocess_img.to(self.device)
            feature_maps = self.feature_extractor(tensor_img).values()
            x_raw_list = []
            for torch_f_map in feature_maps:
                f_map = torch_f_map.detach().cpu().float().numpy()[0]
                f_map = f_map.reshape(f_map.shape[0], f_map.shape[1] * f_map.shape[2])
                p, m = f_map.shape[0], f_map.shape[1]
                mu = np.mean(f_map, axis=1).reshape(-1, 1)
                sigma = (f_map - mu) @ (f_map - mu).T / m
                mu = mu / (p * (p - 1))
                sigma = sigma / (p * (p - 1))
                sigma_flat = sigma.flatten()
                x_raw = np.concatenate([mu, sigma_flat.reshape(-1, 1)])
                x_raw_list.append(x_raw)
        del preprocess_img, tensor_img, feature_maps
        return np.concatenate(x_raw_list).flatten()

    def find_archetypes(self):
        temp_dir = "temp_feature"
        os.makedirs(temp_dir, exist_ok=True)
        features_files = []
        valid_paths = []
        print("Starting transformation of the data...")
        for i, path in enumerate(tqdm(self.data_path)):
            with torch.no_grad():
                img = imread_safe(path)
                # print(f"Image n° {i}, taking features from {path}")
                """This code is taken from features_extractor.py"""
                h, w = img.shape[:2]
                m = max(h, w)
                if m != 1800:
                    continue
                features = self.transform(img)
                temp_file = os.path.join(temp_dir, f"feature_{i}.npy")
                np.save(temp_file, features)
                valid_paths.append(path)
                features_files.append(temp_file)

                del img, features
                if i % 10 == 0:
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
        self.data_path = np.array(valid_paths)
        n_samples = len(features_files)
        n_components = min(512, n_samples - 1)
        ipca = IncrementalPCA(n_components)

        print("Starting IncrementalPCA fitting...")
        for i in tqdm(range(0, n_samples, 512)):
            batch = features_files[i : i + 512]
            if len(batch) < n_components:
                continue
            batch_data = [np.load(f) for f in batch]
            ipca.partial_fit(batch_data)
            del batch_data

        print("Starting transformation of the data via IncrementalPCA...")
        X_transformed_list = []
        for i in tqdm(range(0, n_samples, 512)):
            batch = features_files[i : i + 512]
            batch_data = [np.load(f) for f in batch]
            X_batch_transformed = ipca.transform(batch_data)
            X_transformed_list.append(X_batch_transformed)
            del batch_data
        X = np.vstack(X_transformed_list)
        print(f"shape of X : {X.shape}")
        self.X = X
        self.ipca = ipca
        print("Starting archetype generation...")
        A = self.archetype.fit_transform(X)
        Z = self.archetype.archetypes_
        B = self.archetype.B_
        self.A = A
        self.B = B
        self.Z = Z
        return A, B, Z, ipca

    def classify_soft(self, x):
        img = imread_safe(x)
        h, w = img.shape[:2]
        m = max(h, w)
        if m != 1800:
            raise ValueError("Longest dimension is not of size 1800")
        x_raw = self.transform(img).reshape(1, -1)
        x_raw_reduced = self.ipca.transform(x_raw)
        return self.archetype.transform(x_raw_reduced)

    def classify_hard(self, x):
        return np.argmax(self.classify_soft(x))

    def getClosestPaintingsForArchetype(self, archetype_index, nb_paintings=1):
        return self.data_path[np.argsort(self.B[archetype_index])[::-1][:nb_paintings]]

    def getPaintingsFromArchetype(self, archetype_index, nb_paintings=None):
        A_archetype = self.A[:, archetype_index]
        sorted_indices = np.argsort(A_archetype)[::-1]
        sorted_A = A_archetype[sorted_indices]
        # We suppose a painting should be associated to an archetype if
        # said archetype makes up for the majority of archetypes contributions
        mask = sorted_A > 1 / self.k
        return self.data_path[sorted_indices[mask][:nb_paintings]]


if __name__ == "__main__":
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
    cur_archetype = 0
    for letter in ["a", "b", "c", "d"]:
        """We use 4 archetypes per group of arts starting with the corresponding letter."""
        print(f"The part of the dataset with names starting with {letter}")
        paths = [
            path
            for path in glob("ArtemisArt/**/*.jpg")
            if path.split("/")[1][0] == letter
        ]
        print(f"Il y a {len(paths)} images dans le dataset")

        n_archetype = 8
        a = ArchetypeGenerator(n_archetype, paths, device, feature_extractor)
        A, B, Z, pca_mod = a.find_archetypes()

        # print(f"Voila la forme de A : {A}\n")
        # print(f"Voila la forme de B : {B}\n")
        # print(f"Voila la forme de Z : {Z}\n")

        print("Soft classifier")
        print(a.classify_soft("ArtemisArt/blake - william-blake_1827/blake_1.jpg"))

        print("Closest paintings for archetype")
        print(a.getClosestPaintingsForArchetype(0))

        print("Paintings from archetype")
        print(a.getPaintingsFromArchetype(0))

        # On choisit l'Archétype 0
        Z_arch_0 = Z[0]

        # Et on lance !
        images_synthetisees = [
            synthetiser_archetype(
                Z_archetype=Z[i],
                pca_model=pca_mod,
                extractor=feature_extractor,
                device=device,
                n_iteration=500,
            )
            for i in range(n_archetype)
        ]

        for i in range(n_archetype):
            plt.imshow(images_synthetisees[i])
            plt.axis("off")
            # plt.title(f"L'essence de l'Archétype {i}")
            plt.savefig(
                f"archetypes/archetype_{cur_archetype}.png",
                bbox_inches="tight",
                dpi=300,
            )
            print(f"Image sauvegardée sous le nom 'archetype_{cur_archetype}.png' !")
            cur_archetype += 1

        plt.close()
