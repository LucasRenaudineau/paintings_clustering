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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from archetype_visualisation import *
from tqdm import tqdm


class ArchetypeGenerator:
    """
    A class to generate archetypes solving the archetype optimization problem for style analysis.
    This class implements the methods explained in the paper `Unsupervised Learning of Artistic Styles with
    Archetypal Style Analysis <https://arxiv.org/pdf/1805.11155>`_ and some other features for conveniance.

    Parameters
    ----------
    nb_archetypes (int) : Number of archetypes to generate.
    data_path (list[str]) : List of paths for the images to generate the archetypes from.
    device (torch.device) : The device to run torch features on.
    feature_extractor (torch.fx.GraphModule) : The layers of the VGG-19 to extract features from.

    Attributes
    ----------
    k (int) : Number of archetypes.
    archetype (class AA) : The **_archetypes_** model to solve the optimization problem
    data_path (list[str]) : The list of paths forthe images to generate the archetypes from.
    device (torch.device) : The device to run torch features on.
    feature_extractor (torch.fx.GraphModule) : The layers of the VGG-19 to extract features from.
    scaler (StandardScaler) : The Scaler model to normalize the vectors before the PCA to avoid layer bias.
    ipca (IncrementalPCA) : The IncrementalPCA model to reduce dimensionality on the data vectors.
    X (ndarray of shape (n_samples, 512)) : The matrix obtained from the reduced data vectors after the IncrementalPCA.
    A (ndarray of shape (n_samples, n_archetypes)) : The coefficients such that X = ZA obtained from the optimization problem.
    B (ndarray of shape (n_archetypes, n_samples)) : The coefficients such that Z = XB obtained from the optimization problem.
    Z (ndarray of shape (n_archetypes, 512)) : The archetype matrix.
    """

    def __init__(
        self,
        nb_archetypes: int,
        data_path: list[str],
        device: torch.device,
        feature_extractor: torch.fx.GraphModule,
    ):
        self.k = nb_archetypes
        self.archetype = AA(nb_archetypes)
        self.data_path = np.array(data_path)
        self.device = device
        self.feature_extractor = feature_extractor

    def transform(self, img: np.ndarray):
        """
        This function transforms the input image into a vector whose parameters are means and covariance matrices
        flattened from the feature maps obtained from the VGG-19 model.

        Parameters
        ----------
        img (ndarray) : The image to transform.

        Returns
        -------
        X (ndarray) : The data vector corresponding to the input image in the form of a 1-dimensional array.

        Notes
        -----
        This function implements the method in paper `Unsupervised Learning of Artistic Styles with
        Archetypal Style Analysis <https://arxiv.org/pdf/1805.11155>`_.
        """
        with torch.no_grad():
            # The image is set as a 1800x1800 image through padding
            preprocess_img = preprocessing_image(img)
            tensor_img = preprocess_img.to(self.device)
            feature_maps = self.feature_extractor(tensor_img).values()
            x_raw_list = []

            for torch_f_map in feature_maps:
                f_map = torch_f_map.detach().cpu().float().numpy()[0]
                # The feature maps are seen as matrices (p,m) where p is the number of channels
                # and m is the number of pixels
                f_map = f_map.reshape(f_map.shape[0], f_map.shape[1] * f_map.shape[2])
                p, m = f_map.shape[0], f_map.shape[1]
                mu = np.mean(f_map, axis=1).reshape(-1, 1)
                sigma = (f_map - mu) @ (f_map - mu).T / m
                mu = mu / (p * (p - 1))
                sigma = sigma / (p * (p - 1))
                sigma_flat = sigma.flatten()
                x_raw = np.concatenate([mu, sigma_flat.reshape(-1, 1)])
                x_raw_list.append(x_raw)

        # Free memory usage
        del preprocess_img, tensor_img, feature_maps
        return np.concatenate(x_raw_list).flatten()

    def find_archetypes(self):
        """
        This function solves the archetypal optimisation problem to find archetypes accordingly
        to the data given to the ArchetypeGenerator class.

        Returns
        -------
        A (ndarray of shape (n_samples, n_archetypes)) : The coefficients such that X = ZA.
        B (ndarray of shape (n_archetypes, n_samples)) : The coefficients such that Z = XB.
        Z (ndarray of shape (n_archetypes, 512)) : The archetypes.
        ipca (IncrematalPCA) : The Incremental PCA model used for reduction. It has 512 components.
        scaler (StandardScaler) : The StandardScaler model used to scale the 512 components.

        Notes
        -----
        This function implements the method in paper `Unsupervised Learning of Artistic Styles with
        Archetypal Style Analysis <https://arxiv.org/pdf/1805.11155>`_.
        """
        temp_dir = "temp_feature"
        os.makedirs(temp_dir, exist_ok=True)
        features_files = []
        valid_paths = []
        print("Starting transformation of the data...")
        for i, path in enumerate(tqdm(self.data_path)):
            with torch.no_grad():
                img = imread_safe(path)
                h, w = img.shape[:2]
                m = max(h, w)
                if m != 1800:
                    continue
                features = self.transform(img)
                # We save the features to be used in the Incremental PCA to avoid RAM issues
                temp_file = os.path.join(temp_dir, f"feature_{i}.npy")
                np.save(temp_file, features)
                valid_paths.append(path)
                features_files.append(temp_file)

                # Free memory usage
                del img, features
                if i % 10 == 0:
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
        self.data_path = np.array(valid_paths)
        n_samples = len(features_files)
        batch_size = 256

        n_components = min(batch_size, n_samples - 1)
        ipca = IncrementalPCA(n_components, whiten=True)

        # Fitting the Incremental PCA
        print("Starting IncrementalPCA fitting...")
        for i in tqdm(range(0, n_samples, batch_size)):
            batch = features_files[i : i + batch_size]
            if len(batch) < n_components:
                continue
            batch_data = [np.load(f) for f in batch]
            batch_data_scaled = np.array(batch_data, dtype=np.float32)
            ipca.partial_fit(batch_data_scaled)
            del batch_data, batch_data_scaled
            gc.collect()

        # Transforming the data via Incremental PCA
        print("Starting transformation of the data via IncrementalPCA...")
        X_transformed_list = []
        for i in tqdm(range(0, n_samples, batch_size)):
            batch = features_files[i : i + batch_size]
            batch_data = [np.load(f) for f in batch]
            batch_data_raw = np.array(batch_data, dtype=np.float32)
            X_batch_transformed = ipca.transform(batch_data_raw)
            X_transformed_list.append(X_batch_transformed)
            del batch_data, batch_data_raw
            gc.collect()

        X = np.vstack(X_transformed_list)
        print(f"shape of X : {X.shape}")
        self.X = X
        self.ipca = ipca

        # Solving of the archetypal optimisation problem to generate archetypes
        print("Starting archetype generation...")
        A = self.archetype.fit_transform(X)
        Z = self.archetype.archetypes_
        B = self.archetype.B_
        self.A = A
        self.B = B
        self.Z = Z
        return A, B, Z, ipca

    def classify_soft(self, x: str):
        """
        Computes the A coefficient for a new image from pre-established archetypes.

        Parameters
        ----------
        x (str) : The path for the new image.

        Returns
        -------
        A (ndarray of shape (n_samples, n_archetypes)) : The A coefficient such that x = ZA
        """
        img = imread_safe(x)
        h, w = img.shape[:2]
        m = max(h, w)
        if m != 1800:
            raise ValueError("Longest dimension is not of size 1800")
        x_raw = self.transform(img).reshape(1, -1).astype(np.float32)
        x_raw_reduced = self.ipca.transform(x_raw)
        return self.archetype.transform(x_raw_reduced)

    def classify_hard(self, x: str):
        """
        Select the highest component from the A coefficient such that x=ZA to assign an archetype for a new image x.

        Parameters
        ----------
        x (str) : The new image.

        Returns
        ------
        The archetype whose coefficient in A is the highest.
        """
        return np.argmax(self.classify_soft(x))

    def getClosestPaintingsForArchetype(
        self, archetype_index: int, nb_paintings: int = 1
    ):
        """
        Find the paintings most responsible for the generation of a given archetype.

        Parameters
        ----------
        archetype_index (int) : The index of the archetype in Z.

        nb_paintings (int) : The number of closest paintings we want to find.

        Returns
        -------
        The list of paintings from the original data.
        """
        return self.data_path[np.argsort(self.B[archetype_index])[::-1][:nb_paintings]]

    def getPaintingsFromArchetype(self, archetype_index: int, nb_paintings: int = None):
        """
        Get the closest paintings from an archetype.

        Parameters
        ----------
        archetype_index (int) : The index of the archetype in Z.

        nb_paintings (int) : The number of closest paintings we want to find.

        Returns
        -------
        The list of paintings from the original data.
        """
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
    for letter in ["a"]:
        """We use 4 archetypes per group of arts starting with the corresponding letter."""
        print(f"The part of the dataset with names starting with {letter}")
        paths = [
            path
            for path in glob("ArtemisArt/**/*.jpg")
            if path.split("/")[1][0] == letter
        ]
        paths = paths[:30]
        print(f"Il y a {len(paths)} images dans le dataset")

        n_archetype = 8
        a = ArchetypeGenerator(n_archetype, paths, device, feature_extractor)
        A, B, Z, pca_mod, scaler = a.find_archetypes()

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
        # Z_arch_0 = Z[0]

        # # Et on lance !
        # images_synthetisees = [
        #     synthetiser_archetype(
        #         Z_archetype=Z[i],
        #         pca_model=pca_mod,
        #         scaler_model=scaler,
        #         extractor=feature_extractor,
        #         device=device,
        #         n_iteration=500,
        #     )
        #     for i in range(1)
        # ]

        # for i in range(1):
        #     plt.imshow(images_synthetisees[i])
        #     plt.axis("off")
        #     # plt.title(f"L'essence de l'Archétype {i}")
        #     plt.savefig(
        #         f"archetypes/archetype_{cur_archetype}.png",
        #         bbox_inches="tight",
        #         dpi=300,
        #     )
        #     print(f"Image sauvegardée sous le nom 'archetype_{cur_archetype}.png' !")
        #     cur_archetype += 1
        print("Starting UMAP visualisation...")
        visualisation_2d(a.X, Z, a.data_path)

        plt.close()
