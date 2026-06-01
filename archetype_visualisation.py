import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# VGG19-BN tapped nodes and their channel counts:
#
#   features.2  → ReLU after Conv(3→64)    → 64 channels
#   features.5  → ReLU after Conv(64→64)   → 64 channels
#   features.9  → ReLU after Conv(64→128)  → 128 channels
#   features.12 → ReLU after Conv(128→128) → 128 channels
#   features.16 → ReLU after Conv(128→256) → 256 channels
#
# Total raw descriptor length: 64+64²+64+64²+128+128²+128+128²+256+256² = 107 136
# ---------------------------------------------------------------------------
VGG19BN_P_LIST = [64, 64, 128, 128, 256]

# ImageNet normalisation constants (must match preprocessing_image in preprocessing.py)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def unpack_Z_full(Z_full_numpy, p_list, device):
    """
    Splits the flat inverse-PCA vector back into per-layer (mu, sigma) targets.

    Parameters
    ----------
    Z_full_numpy : np.ndarray, shape (D,)
        The archetype vector projected back to the raw descriptor space via
        pca_model.inverse_transform().  D must equal sum(p + p² for p in p_list).
    p_list : list[int]
        Number of channels at each tapped VGG layer, in order.
        For the nodes used here: [64, 64, 128, 128, 256].
    device : torch.device

    Returns
    -------
    list of dict {"mu": Tensor(p,1), "sigma": Tensor(p,p)}
    """
    expected_len = sum(p + p * p for p in p_list)
    if len(Z_full_numpy) != expected_len:
        raise ValueError(
            f"Z_full_numpy has length {len(Z_full_numpy)} but p_list={p_list} "
            f"requires {expected_len} values. "
            f"Check that p_list matches the tapped VGG nodes used in transform()."
        )

    cibles = []
    idx = 0

    for p in p_list:
        mu_flat    = Z_full_numpy[idx : idx + p]
        idx       += p
        sigma_flat = Z_full_numpy[idx : idx + p * p]
        idx       += p * p

        mu    = torch.tensor(mu_flat,    dtype=torch.float32, device=device).view(p, 1)
        sigma = torch.tensor(sigma_flat, dtype=torch.float32, device=device).view(p, p)

        cibles.append({"mu": mu, "sigma": sigma})

    return cibles


def total_variation_loss(img):
    """
    Anisotropic total-variation loss on a (1, C, H, W) image tensor.
    Encourages local smoothness (brush-stroke-like texture) without blurring
    across long distances.
    """
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :]  - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:]  - img[:, :, :, :-1]))
    return tv_h + tv_w


def calculer_mu_sigma_pytorch(torch_f_map):
    """
    Computes the normalised first- and second-order statistics of a feature map,
    matching the formula used in ArchetypeGenerator.transform().

    Parameters
    ----------
    torch_f_map : Tensor, shape (1, p, H, W)   [batch dim included]

    Returns
    -------
    mu    : Tensor, shape (p, 1)
    sigma : Tensor, shape (p, p)
    Both normalised by p*(p-1) as in the paper.
    """
    _, p, h, w = torch_f_map.shape
    m = h * w

    # (p, H*W) — drop batch dim first to avoid silent reshape errors
    f_map_flat = torch_f_map.squeeze(0).view(p, m)

    mu = torch.mean(f_map_flat, dim=1).view(p, 1)

    f_map_centered = f_map_flat - mu
    sigma = torch.mm(f_map_centered, f_map_centered.t()) / m

    # Normalisation from the paper (prevents high-channel layers from dominating)
    norm = p * (p - 1)
    mu    = mu    / norm
    sigma = sigma / norm

    return mu, sigma


def synthetiser_archetype(
    Z_archetype,
    pca_model,
    extractor,
    device,
    n_iteration,
    p_list=VGG19BN_P_LIST,
    image_size=224,
    lr=0.05,
    poids_style=1e6,
    poids_tv=1e-3,
):
    """
    Synthesises a texture image whose deep style statistics match archetype Z_archetype.

    The optimisation minimises:
        L = poids_style * sum_l [ MSE(mu_l, mu_l*) + MSE(sigma_l, sigma_l*) ]
          + poids_tv   * TV(image)

    where (mu_l*, sigma_l*) are the target statistics decoded from Z_archetype.

    Parameters
    ----------
    Z_archetype  : np.ndarray, shape (n_components,)
        One row of the AA archetype matrix Z (in PCA-reduced space).
    pca_model    : sklearn PCA
        The fitted PCA used during find_archetypes(); used to invert Z back to
        the raw descriptor space.
    extractor    : nn.Module
        The tapped VGG feature extractor (create_feature_extractor output).
    device       : torch.device
    n_iteration  : int
        Number of gradient-descent steps.
    p_list       : list[int]
        Channel counts at each tapped layer. Must match what transform() used.
        Default: VGG19BN_P_LIST = [64, 64, 128, 128, 256].
    image_size   : int
        Spatial resolution of the synthesised image (square). 224 is fast;
        512 gives richer textures but uses more VRAM.
    lr           : float
        Adam learning rate.
    poids_style  : float
        Weight on the style loss.
    poids_tv     : float
        Weight on the total-variation regulariser. Set to 0 to disable (expect
        noisy results). A value around 1e-3 works well in practice.

    Returns
    -------
    np.ndarray, shape (image_size, image_size, 3), dtype float32, values in [0, 1]
    """
    # --- 1. Decode archetype from PCA space to raw descriptor space -----------
    Z_full_numpy = pca_model.inverse_transform(Z_archetype.reshape(1, -1))[0]

    # --- 2. Split into per-layer (mu, sigma) targets -------------------------
    cibles_couches = unpack_Z_full(Z_full_numpy, p_list, device)

    # --- 3. Initialise the image to optimise ---------------------------------
    # Random initialisation in [0, 1]; requires_grad so Adam can update it.
    image_generee = torch.rand(
        1, 3, image_size, image_size, device=device, requires_grad=True
    )
    optimizer = optim.Adam([image_generee], lr=lr)

    print(f"Début de la synthèse ({n_iteration} itérations, "
          f"image {image_size}x{image_size})...")

    # --- 4. Optimisation loop ------------------------------------------------
    for iteration in range(n_iteration):
        optimizer.zero_grad()

        # Normalise with ImageNet stats so VGG sees the expected colour range.
        # TF.normalize expects (C, H, W), so we squeeze/unsqueeze the batch dim.
        img_norm = TF.normalize(
            image_generee.squeeze(0),
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
        ).unsqueeze(0)

        features_actuelles_dict = extractor(img_norm)

        # Style loss: sum of MSE on mu and sigma across all layers
        loss_style = torch.tensor(0.0, device=device)
        for i, torch_f_map in enumerate(features_actuelles_dict.values()):
            mu_actuel, sigma_actuel = calculer_mu_sigma_pytorch(torch_f_map)
            cible = cibles_couches[i]
            loss_style = loss_style + F.mse_loss(mu_actuel,    cible["mu"])    * poids_style
            loss_style = loss_style + F.mse_loss(sigma_actuel, cible["sigma"]) * poids_style

        # TV regularisation (smoothness prior)
        loss_tv = total_variation_loss(image_generee)

        loss_totale = loss_style + poids_tv * loss_tv
        loss_totale.backward()
        optimizer.step()

        # Keep pixel values in the valid display range
        image_generee.data.clamp_(0.0, 1.0)

        if iteration % 100 == 0:
            print(f"  Itération {iteration:03d} | "
                  f"Loss style: {loss_style.item():.2f} | "
                  f"Loss TV: {loss_tv.item():.4f}")

    # --- 5. Return as a numpy HWC array in [0, 1] ----------------------------
    image_finale = (
        image_generee.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    )
    return image_finale