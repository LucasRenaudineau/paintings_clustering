# paintings_clustering

Telecom Paris IMA project of clustering unlabelled paintings images.

# Structure

## Images

Paintings are situated at ./ArtemisArt/

## Models

Models are loaded at ./models/

## Outputs

Outputs for examples and showcases are sent to ./outputs/

# Methods

## 1st method : Archetypes method

Files used for this method are:

- preprocessing.py (preprocesses the image into a 1800*1800 image mirrored, then produces two images : downscaled image 224*224, most uniform crop of size 224*224 grayish)
- archetype_visualisation.py (implements two features : one to visualize archetypes as images to see the proximity to the paintnigs and the other to project archetypes and paintings on a 2D plane using UMAP to have another way to verify similarities)
- archetype_clustering.py (implements clustering on the archetypes to then classify paintings)
- archetype_style_analysis.py (generates archetypes using the pipline developped in the paper [Unsupervised Learning of Artistic Styles with Archetypal Style Analysis](https://arxiv.org/pdf/1805.11155), it also has some tests at the end of the file with __name__ = '__main__' to try the methods)
- main_archetype.ipynb (notebook that uses every other python file to test the full pipeline)

## 2nd method : Graph method

Files used for this method are:

- invalid_size.py (deletes images that do not have a max dimension of 1800 pixels)
- preprocessing.py (preprocesses the image into a 1800*1800 image mirrored, then produces two images : downscaled image 224*224, most uniform crop of size 224*224 grayish)
- features_extractor.py (retrieves the features of an image: the 5 convolutional activation layers of the downscaled image passed into resnet, and the 5 convolutional activation layers of the crop passed into resnet)
- features_comparator.py (implements a weighted cossine dissimilarity distance measure between to sets of features)
- knn.py (creates a non-complete graph where each image is a node and has only k neighbors which are expected to be its closest neighbors in the theorical complete graph of distances)
- hdbscan_clustering.py (uses hdbscan to compute classes based on the non-complete graph computed in knn)

Each function has some tests in if __name__=='__main__' section. To try the full method, use python hdbscan_clustering.py and wait. The classes are outputed in ./outputs/hdbscan_classes/classx/
