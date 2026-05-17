import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input



INPUT_SHAPE = (224, 224) # ResNet50 input size


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


# Image preprocessing
 
def resize_image(image: np.ndarray) -> np.ndarray:
    """
    Pad then resize a BGR image to the 224x224 input size expected by ResNet50.
 
    Args:
        image: NumPy array of shape (H, W, 3), BGR uint8.
               The longest dimension must be 1800px.
 
    Returns:
        NumPy array of shape (1, 224, 224, 3), float32, ready for inference.
    """
    squared = pad_to_square(image)
    rgb = squared[:, :, ::-1]      # Convert from BGR to RGB (::-1 to read in the opposite direction)
    
    # IS TENSORFLOW THE BEST FUNCTION ? 
    resized = tf.image.resize(rgb, INPUT_SHAPE).numpy().astype(np.float32) # resized to 224*224
 
    batched = np.expand_dims(resized, axis=0) # adding batch dimension : (1, 224, 224, 3)
    return preprocess_input(batched) # applying resnet50 preprocessing





if __name__ == "__main__":
    img = imread_safe("ArtemisArt/afro - afro-basaldella_1912/afro_1.jpg")
    squared_image = pad_to_square(img)
    cv2.imwrite("outputs/afro_1_squared.jpg", squared_image)
    
    img = imread_safe("ArtemisArt/bernard - emile-bernard_1868/bernard_21.jpg")
    squared_image = pad_to_square(img)
    cv2.imwrite("outputs/bernard_21.jpg", squared_image)

