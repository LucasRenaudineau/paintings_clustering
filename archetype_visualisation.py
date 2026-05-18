def unpack_Z_full(Z_full_numpy, p_list, device):
    """
    p_list contient le nombre de canaux de tes 5 couches VGG.
    Ex: pour vgg19_bn (features 2, 5, 9, 12, 16), c'est [64, 64, 128, 128, 256]
    """
    cibles = []
    idx = 0

    for p in p_list:
        taille_mu = p
        taille_sigma = p * p

        # Découpage du tableau NumPy
        mu_flat = Z_full_numpy[idx : idx + taille_mu]
        idx += taille_mu

        sigma_flat = Z_full_numpy[idx : idx + taille_sigma]
        idx += taille_sigma

        # Conversion en Tenseurs PyTorch sur la carte graphique
        mu = torch.tensor(mu_flat, dtype=torch.float32, device=device).view(p, 1)
        sigma = torch.tensor(sigma_flat, dtype=torch.float32, device=device).view(p, p)

        cibles.append({"mu": mu, "sigma": sigma})

    return cibles


import torch.nn.functional as F


def total_variation_loss(img):
    # Calcule la différence entre les pixels voisins pour forcer des "coups de pinceau"
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


def calculer_mu_sigma_pytorch(torch_f_map):
    # La shape est (1, p, h, w) car il y a la dimension du batch
    p, h, w = torch_f_map.shape[1], torch_f_map.shape[2], torch_f_map.shape[3]
    m = h * w

    # On aplatit les dimensions spatiales
    f_map_flat = torch_f_map.view(p, m)

    # Moyenne (mu)
    mu = torch.mean(f_map_flat, dim=1).view(p, 1)

    # Covariance (sigma)
    f_map_centered = f_map_flat - mu
    sigma = torch.mm(f_map_centered, f_map_centered.t()) / m

    # Normalisation de ton papier
    mu = mu / (p * (p - 1))
    sigma = sigma / (p * (p - 1))

    return mu, sigma


import torch.optim as optim


def synthetiser_archetype(
    Z_archetype, pca_model, extractor, device, p_list=[64, 64, 128, 128, 256]
):

    Z_full_numpy = pca_model.inverse_transform(Z_archetype.reshape(1, -1))[0]
    cibles_couches = unpack_Z_full(Z_full_numpy, p_list, device)

    image_generee = torch.rand(1, 3, 224, 224, device=device, requires_grad=True)

    # On baisse légèrement le learning rate pour éviter que l'IA ne panique
    optimizer = optim.Adam([image_generee], lr=0.05)

    # L'outil de normalisation obligatoire pour que le VGG "voie" bien les couleurs
    # ... (début de la fonction inchangé) ...
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    print("Début de la peinture par IA...")

    # LES DEUX LIGNES MAGIQUES MANQUANTES
    poids_style = 1e6
    poids_tv = 0.0  # On désactive complètement le lissage pour l'instant

    for iteration in range(1000):
        optimizer.zero_grad()

        img_norm = normalize(image_generee)
        features_actuelles_dict = extractor(img_norm)

        loss_style = 0
        for i, torch_f_map in enumerate(features_actuelles_dict.values()):
            mu_actuel, sigma_actuel = calculer_mu_sigma_pytorch(torch_f_map)
            cible = cibles_couches[i]

            # ON APPLIQUE LE MULTIPLICATEUR ICI !
            loss_style += F.mse_loss(mu_actuel, cible["mu"]) * poids_style
            loss_style += F.mse_loss(sigma_actuel, cible["sigma"]) * poids_style

        loss_tv = total_variation_loss(image_generee)

        # ON UTILISE LES NOUVEAUX POIDS
        loss_totale = loss_style + (poids_tv * loss_tv)

        loss_totale.backward()
        optimizer.step()

        image_generee.data.clamp_(0, 1)

        if iteration % 100 == 0:
            print(f"Itération {iteration:03d} | Loss: {loss_totale.item():.2f}")

    image_finale = image_generee.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return image_finale
