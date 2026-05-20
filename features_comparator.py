from preprocessing import *
from features_extractor import *

# Distance function between 2 activations
def distance(activations1, activations2):
    pass

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
