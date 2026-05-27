from preprocessing import *
from features_extractor import *
import os
from pathlib import Path


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
        2 Arrays mapping each layer name in order to its activation NumPy array.
            0  -> (1, 112, 112, 64)      (conv1_relu)
            1  -> (1, 56, 56, 256)       (conv2_block3_out)
            2  -> (1, 28, 28, 512)       (conv3_block4_out)
            3  -> (1, 14, 14, 1024)      (conv4_block6_out)
            4  -> (1, 7, 7, 2048)        (conv5_block3_out)
        
    Returns:
        float : value of the distance measure between features1 and features2  
    """
    
    dist = 0
    dist += cosineSimilarity(features1[0],features2[0])
    dist += 0.1*cosineSimilarity(features1[1],features2[1])
    dist += 0.1*cosineSimilarity(features1[2],features2[2])
    dist += cosineSimilarity(features1[3],features2[3])
    dist += cosineSimilarity(features1[4],features2[4])
    
    return dist

# Test function to compute distances between an image and a list of images
def compute_distances_one_to_many(image_path,comparedImagesPaths, feature_model):
    image = imread_safe(image_path)
    comparedImages = [imread_safe(path) for path in comparedImagesPaths]
    
    activation_set = extract_features(image, feature_model)
    compared_activations_sets = [extract_features(im, feature_model) for im in comparedImages]

    distances = []

    for i in range(len(comparedImagesPaths)):
        print(f"Distance between {image_path} and {comparedImagesPaths[i]} is:")
        dist = distance(activation_set, compared_activations_sets[i])
        print(dist)
        distances.append(dist)
    return distances

def find_paths(n:int):
    """ 
    Keep the n first images in dataset.
    14553 images maximum
    
    Args:
        n : number of images to keep
        
    Returns:
        numpy Array of string : path from starting with ArtemisArt folder
    """
    dir = Path("ArtemisArt/")
    imagesPath = dir.rglob("*.jpg")
    imagesPath = np.array(list(imagesPath))[:n]     # Because imagesPath is a "map object", I am obliged to transform into list then np array... 
    imagesPath = [str(p) for p in imagesPath]       # Don't know if needed, but I prefer not work with "PosixPath" objects
    #print(imagesPath)

    return imagesPath

# Save the n nearest and n furthest images from the dataset
def save_nearest_and_furthest_images(image_path, comparedImagesPaths, n, feature_model):
    distances = compute_distances_one_to_many(image_path, comparedImagesPaths, feature_model)
    np_distances = np.array(distances)
    min_indices = np.argpartition(np_distances, n-1)[:n]
    sorted_min_indices = min_indices[np.argsort(np_distances[min_indices])]
    max_indices = np.argpartition(np_distances, -n)[-n:]
    sorted_max_indices = max_indices[np.argsort(np_distances[max_indices])[::-1]]

    for i in range(n):
        min_index = sorted_min_indices[i]
        max_index = sorted_max_indices[i]
        img_min = imread_safe(comparedImagesPaths[min_index])
        cv2.imwrite("outputs/n_min/"+comparedImagesPaths[min_index].split("/")[-1], img_min)
        print(f"In n_min : {comparedImagesPaths[min_index]} with a score of {distances[min_index]}.")

        img_max = imread_safe(comparedImagesPaths[max_index])
        cv2.imwrite("outputs/n_max/"+comparedImagesPaths[max_index].split("/")[-1], img_max)
        print(f"In n_max : {comparedImagesPaths[max_index]} with a score of {distances[max_index]}.")

if __name__ =="__main__":
    _, feature_model = load_model()
    paths = find_paths(100)
    print("Image to compare is " + paths[0])
    save_nearest_and_furthest_images(paths[0], paths, 20, feature_model)
