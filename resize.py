import cv2
import numpy as np

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

    if h == w:
        return image

    if h != target and w != target:
        raise ValueError(f"Expected the longest dimension to be {target}, got shape {h}x{w}.")

    if w < h:
        # Portrait: copy leftmost (target - w) columns and append on the right
        pad = target - w
        fill = image[:, :pad]
        return np.concatenate([image, fill], axis=1)
    else:
        # Landscape: copy topmost (target - h) rows and append at the bottom
        pad = target - h
        fill = image[:pad, :]
        return np.concatenate([image, fill], axis=0)

if __name__ == "__main__":
    img = imread_safe("ArtemisArt/afro - afro-basaldella_1912/afro_1.jpg")
    squared_image = pad_to_square(img)
    cv2.imwrite("outputs/afro_1_squared.jpg", squared_image)
