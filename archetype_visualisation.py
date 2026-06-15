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
_IMAGENET_STD = [0.229, 0.224, 0.225]


def unpack_Z_full(Z_full_numpy, p_list, device):
    """
    Splits the flat inverse-PCA vector back into per-layer (mu, sigma) targets,
    and **reverses the p*(p-1) normalisation** so the targets are in raw feature
    space (same scale as the statistics computed by calculer_mu_sigma_pytorch).

    Background: in ArchetypeGenerator.transform(), mu and sigma at each layer are
    divided by p*(p-1) before being stored in X.  This normalisation is useful for
    PCA/AA (it prevents high-channel layers from dominating the variance), but it
    shrinks all values to ~1e-4.  If the targets kept that scale, the MSE loss
    would start at ~1e-8 and vanish in a few iterations regardless of the image.
    Multiplying back by p*(p-1) restores values to their natural range (~0.1–1.0),
    giving a meaningful, slowly-decreasing style loss throughout optimisation.

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
        Targets in raw (un-normalised) feature space.
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
        norm = p * (p - 1)  # inverse of the normalisation applied in transform()

        mu_flat = Z_full_numpy[idx : idx + p]
        idx += p
        sigma_flat = Z_full_numpy[idx : idx + p * p]
        idx += p * p

        # Multiply back by p*(p-1) to undo the normalisation stored in X/Z
        mu = torch.tensor(mu_flat * norm, dtype=torch.float32, device=device).view(p, 1)
        sigma = torch.tensor(
            sigma_flat * norm, dtype=torch.float32, device=device
        ).view(p, p)

        cibles.append({"mu": mu, "sigma": sigma})

    return cibles


def total_variation_loss(img):
    """
    Anisotropic total-variation loss on a (1, C, H, W) image tensor.
    Encourages local smoothness (brush-stroke-like texture) without blurring
    across long distances.
    """
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w


def calculer_mu_sigma_pytorch(torch_f_map):
    """
    Computes the raw first- and second-order statistics of a feature map.

    Intentionally does NOT apply the p*(p-1) normalisation used in
    ArchetypeGenerator.transform().  That normalisation is only needed when
    building the PCA/AA descriptor (to prevent high-channel layers from
    dominating the variance).  Here, both the computed statistics and the
    targets from unpack_Z_full() are in raw space, so the MSE is meaningful
    and has the right scale for optimisation (~0.1–10 per layer before weighting).

    Parameters
    ----------
    torch_f_map : Tensor, shape (1, p, H, W)   [batch dim included]

    Returns
    -------
    mu    : Tensor, shape (p, 1)   — mean activation per channel
    sigma : Tensor, shape (p, p)   — channel covariance matrix
    """
    _, p, h, w = torch_f_map.shape
    m = h * w

    # (p, H*W) — drop batch dim first to avoid silent reshape errors
    f_map_flat = torch_f_map.squeeze(0).view(p, m)

    mu = torch.mean(f_map_flat, dim=1).view(p, 1)

    f_map_centered = f_map_flat - mu
    sigma = torch.mm(f_map_centered, f_map_centered.t()) / m

    return mu, sigma


def synthetiser_archetype(
    Z_archetype,
    pca_model,
    extractor,
    device,
    n_iteration,
    p_list=VGG19BN_P_LIST,
    image_size=224,
    lr=0.01,
    poids_style=[1.0, 0.8, 0.5, 0.3, 0.1],
    poids_tv=0.1,
    log_every=100,
):
    """
    Synthesises a texture image whose deep style statistics match archetype Z_archetype.

    The optimisation minimises:
        L = poids_style * sum_l [ MSE(mu_l, mu_l*) + MSE(sigma_l, sigma_l*) ]
          + poids_tv   * TV(image)

    where (mu_l*, sigma_l*) are the target statistics decoded from Z_archetype,
    both in raw (un-normalised) feature space.

    --- Tuning guide ---

    The system has 224*224*3 = 150 528 pixel parameters and ~107 136 style
    constraints, so the style loss can reach near-zero quickly.  The key is to
    keep poids_tv high enough so that TV regularisation shapes the image *while*
    the style loss is still non-trivial:

      poids_style=1.0, poids_tv=0.1, lr=0.01   ← recommended defaults
        Style and TV contribute roughly equally at initialisation, giving a
        well-textured result.  Increase poids_tv toward 1.0 for smoother images;
        decrease toward 0.01 for more detail/noise.

      poids_tv=1e-3  (old default)
        TV contributes < 0.1% of the total loss → image converges to an
        arbitrary solution that satisfies the stats but looks like noise.

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
        Number of gradient-descent steps.  500 is a minimum; 1000–2000 gives
        better convergence with the lower lr.
    p_list       : list[int]
        Channel counts at each tapped layer. Must match what transform() used.
        Default: VGG19BN_P_LIST = [64, 64, 128, 128, 256].
    image_size   : int
        Spatial resolution of the synthesised image (square). 224 is fast;
        512 gives richer textures but uses more VRAM.
    lr           : float
        Adam learning rate.  0.01 lets style and TV losses co-evolve rather
        than letting style collapse instantly.
    poids_style  : float
        Weight on the style loss.
    poids_tv     : float
        Weight on the total-variation regulariser.  Must be comparable to
        poids_style (same order of magnitude) to meaningfully shape the image.
        Rule of thumb: poids_tv * TV_init ≈ 0.3 * poids_style * style_init.
    log_every    : int
        Print diagnostics every this many iterations.

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

    print(
        f"Début de la synthèse ({n_iteration} itérations, "
        f"image {image_size}x{image_size}, lr={lr}, "
        f"poids_style={poids_style}, poids_tv={poids_tv})..."
    )

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
            loss_style = (
                loss_style + F.mse_loss(mu_actuel, cible["mu"]) * poids_style[i]
            )
            loss_style = (
                loss_style + F.mse_loss(sigma_actuel, cible["sigma"]) * poids_style[i]
            )

        # TV regularisation (smoothness prior)
        loss_tv = total_variation_loss(image_generee)

        loss_totale = loss_style + poids_tv * loss_tv
        loss_totale.backward()
        optimizer.step()

        # Keep pixel values in the valid display range
        image_generee.data.clamp_(0.0, 1.0)

        if iteration % log_every == 0:
            # Use scientific notation so near-zero values are visible (not "0.00")
            print(
                f"  Itération {iteration:04d} | "
                f"Style: {loss_style.item():.4e} | "
                f"TV: {loss_tv.item():.4e} | "
                f"Total: {loss_totale.item():.4e}"
            )

    # --- 5. Return as a numpy HWC array in [0, 1] ----------------------------
    image_finale = image_generee.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return image_finale
