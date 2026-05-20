from preprocessing import *
from features_extractor import *


def cosineSimilarity(activation1, activation2):
    """ 
    Implement the cosine similarity norm.
    Source : https://arxiv.org/pdf/2407.08623
             IA used to give the idea to linearise the numpy arrays
    
    Args:
        2 numpy of size (1, H, W, C)
    
    Returns:
        float : value of the cosine similarity measure between activation1 and activation2
            => between 0 and 1
    """
    
    act1Norm = np.linalg.norm(activation1)
    act2Norm = np.linalg.norm(activation2)
    
    act1Flat = activation1.ravel()
    act2Flat = activation2.ravel()
    
    return (1-np.dot(act1Flat,act2Flat)/(act1Norm*act2Norm))/2
    
    
    

# Distance function between 2 activations
def distance(features1, features2):
    """ 
    Compute the norm with every features of 2 images.
    For now, it is a sum of every cosineSimilarity of each activation.
    
    Args:
        2 Dictionaries mapping each layer name to its activation NumPy array.
            conv1_relu                -> (1, 112, 112, 64)
            conv2_block3_out          -> (1, 56, 56, 256)
            conv3_block4_out          -> (1, 28, 28, 512)
            conv4_block6_out          -> (1, 14, 14, 1024)
            conv5_block3_out          -> (1, 7, 7, 2048)
        
    Returns:
        float : value of the distance measure between features1 and features2  
    """
    
    # temporary, we will edit the code later
    dist = 0
    dist += cosineSimilarity(features1["conv1_relu"],features2["conv1_relu"])
    dist += cosineSimilarity(features1["conv2_block3_out"],features2["conv2_block3_out"])
    dist += cosineSimilarity(features1["conv3_block4_out"],features2["conv3_block4_out"])
    dist += cosineSimilarity(features1["conv4_block6_out"],features2["conv4_block6_out"])
    dist += cosineSimilarity(features1["conv5_block3_out"],features2["conv5_block3_out"])
    
    return dist

# Test function to compute distances between an image and a list of images
def compute_distances_one_to_many(image_path,comparedImagesPaths, feature_model):
    image = imread_safe(image_path)
    comparedImages = [imread_safe(path) for path in comparedImagesPaths]
    
    activation_set = extract_features(image, feature_model)
    compared_activations_sets = [extract_features(im, feature_model) for im in comparedImages]

    for i in range(len(comparedImagesPaths)):
        print(f"Distance between {image_path} and {comparedImagesPaths[i]} is:")
        print(distance(activation_set, compared_activations_sets[i]))

if __name__ =="__main__":
    _, feature_model = load_model()
    image_path = "ArtemisArt/braque - georges-braque_1882/braque_1.jpg"
    comparedImagesPaths = ["ArtemisArt/braque - georges-braque_1882/braque_1.jpg", "ArtemisArt/braque - georges-braque_1882/braque_2.jpg", "ArtemisArt/braque - georges-braque_1882/braque_3.jpg", "ArtemisArt/braque - georges-braque_1882/braque_4.jpg", "ArtemisArt/braque - georges-braque_1882/braque_5.jpg", "ArtemisArt/braque - georges-braque_1882/braque_6.jpg", "ArtemisArt/braque - georges-braque_1882/braque_7.jpg", "ArtemisArt/brauner - victor-brauner_1903/brauner_1.jpg", "ArtemisArt/bronzino - agnolo-di-cosimi-dit-bronzino_1503/bronzino_1.jpg", "ArtemisArt/brueghel-j - jan-brueghel-l-ancien-dit-de-velours_1568/brueghel-j-I_1.jpg", "ArtemisArt/delacroix - eugene-delacroix_1798/delacroix_33.jpg"]
    compute_distances_one_to_many(image_path, comparedImagesPaths, feature_model)
