import cv2
import numpy as np
from torchvision import transforms, models



INPUT_SHAPE = (224, 224) # ResNet50 input size
CROP_SIZE = 256          # Size of the "most uniform region" crop fed to ResNet


def imread_safe(path: str) -> np.ndarray:
    """cv2.imread replacement that handles spaces and unicode in paths."""
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def pad_to_square(image: np.ndarray) -> np.ndarray:
    """
    Pad an image to 1800x1800 by copying pixels:
      - portrait  (w < h): copy the left strip and paste on the right
      - landscape (h < w): copy the top strip and paste at the bottom
    Assumes the longest dimension is already 1800px.
    """
    h, w = image.shape[:2]
    target = 1800

    if h != target and w != target:
        raise ValueError(f"Expected the longest dimension to be {target}, got shape {h}x{w}.")

    if h == w:
        return image

    if w < h:
        # Portrait: copy leftmost (target - w) columns and append on the right
        pad = target - w
        return cv2.copyMakeBorder(image, 0,0,0,pad,borderType=cv2.BORDER_REFLECT101)    # 101 for "don't repeat the pixel at the border
    else:
        # Landscape: copy topmost (target - h) rows and append at the bottom
        pad = target - h
        return cv2.copyMakeBorder(image, 0,pad,0,0,borderType=cv2.BORDER_REFLECT101) 


# Most-uniform region crop

def find_most_uniform_crop(image: np.ndarray, crop_size: int = CROP_SIZE) -> np.ndarray:
    """
    Find the most uniform (lowest-variance) square window of the image.

    The local variance of every candidate top-left position is computed in O(1)
    via integral images of the grayscale intensity (sum and sum-of-squares), so
    the whole search is a couple of vectorised array ops regardless of image size.
    The window with the smallest variance is the "flattest" region — typically a
    sky, wall or uniformly painted background — and is returned as a BGR crop.

    This runs on the *original* (un-padded) image so the reflected padding added
    by pad_to_square() can never be selected as a (trivially) uniform region.

    Args:
        image    : NumPy array of shape (H, W, 3), BGR uint8.
        crop_size: side length of the square window to extract.

    Returns:
        BGR uint8 crop of shape (crop_size, crop_size, 3). If the image is
        smaller than crop_size in any dimension, the whole image is resized up.
    """
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        return cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    # Integral images: S[y, x] / S2[y, x] = sum / sum-of-squares of gray[:y, :x].
    S, S2 = cv2.integral2(gray)  # both shape (h+1, w+1)

    cs = crop_size
    # Window sum over every top-left position via the inclusion-exclusion formula.
    win_sum  = (S[cs:, cs:]  - S[:-cs, cs:]  - S[cs:, :-cs]  + S[:-cs, :-cs])
    win_sum2 = (S2[cs:, cs:] - S2[:-cs, cs:] - S2[cs:, :-cs] + S2[:-cs, :-cs])

    area = cs * cs
    variance = win_sum2 / area - (win_sum / area) ** 2  # E[X^2] - E[X]^2
    y, x = np.unravel_index(np.argmin(variance), variance.shape)

    return image[y:y + cs, x:x + cs]


# Image preprocessing

def preprocessing_image(image: np.ndarray, pad: bool = True) -> np.ndarray:
    """
    Pad (optionally) then resize a BGR image to the 224x224 input size.
    Then, apply the preprocessing needed for ResNet50.

    Args:
        image: NumPy array of shape (H, W, 3), BGR uint8.
               When pad=True the longest dimension must be 1800px.
        pad  : if True, pad to square with pad_to_square() before resizing.
               Set to False for already-square inputs such as the uniform crop.

    Returns:
       Tensor array of shape (1, 224, 224, 3), float32
    """
    squared = pad_to_square(image) if pad else image
    rgb = squared[:, :, ::-1]      # Convert from BGR to RGB (::-1 to read in the opposite direction)

    # Resize to (224,224)
    resized = cv2.resize(rgb, INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)

    # applying resnet50 preprocessing => make a tensor [(224,224,3) -> (3,224,224)], then normalize with ResNet normalization values
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])        # Used AI for this preprocessing
    
    tensor = transform(resized)
    return tensor.unsqueeze(0) # adding batch dimension : (1, 3, 224, 224)





if __name__ == "__main__":
    
    # Pad tests
    
    img = imread_safe("ArtemisArt/afro - afro-basaldella_1912/afro_1.jpg")
    squared_image = pad_to_square(img)
    cv2.imwrite("outputs/afro_1_squared.jpg", squared_image)
    
    img = imread_safe("ArtemisArt/bernard - emile-bernard_1868/bernard_21.jpg")
    squared_image = pad_to_square(img)
    cv2.imwrite("outputs/bernard_21.jpg", squared_image)

