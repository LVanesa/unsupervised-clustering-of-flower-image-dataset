# Unsupervised Flower Image Clustering

Project for the Practical Machine Learning course (MSc Data Science, Year I, Semester I, University of Bucharest).

## Task Description

Group flower images into classes **without using labels**, then measure how close the clusters come to the real classes.

**Data:** 1,652 images of roses, sunflowers and tulips from a web-scraped Kaggle dataset, cleaned by hand. All images were center-cropped and resized to 256×256. The split is a stratified 70/15/15 train / validation / test.

## The solution

I compared two handcrafted feature representations and two clustering algorithms:

| | Feature Set 1 | Feature Set 2 |
|---|---|---|
| **Shape / texture** | HOG + LBP | Dense SIFT → Bag of Visual Words (250-word vocabulary built with MiniBatch K-Means) with TF-IDF weighting |
| **Color** | HSV histograms | HSV histograms |
| **Combination** | Weighted concatenation (60% HOG / 20% LBP / 20% color) | L2-normalized concatenation |

- **Clustering:** BIRCH and K-Medoids. K-Medoids was tested with Euclidean, cosine and Manhattan distances.
- **Dimensionality reduction:** PCA with 30 to 150 components, tuned together with each algorithm's hyperparameters.
- **Evaluation:** cluster accuracy (the Hungarian algorithm matches each cluster to a class), ARI, NMI and Silhouette. Configurations were chosen by ARI on the validation set.
- **Baselines:** a random classifier as the lower bound and a tuned RBF SVM as the supervised upper bound.

## Results (test set)

| Method | Feature Set 1 | Feature Set 2 |
|---|---|---|
| Random baseline | 31.9% | 31.9% |
| BIRCH | 58.1% | **63.7%** |
| K-Medoids | 56.0% | 61.3% |
| SVM (supervised) | 75.0% | 85.5% |

The richer Dense SIFT + BoVW features improved every model. Sunflowers formed the purest cluster. Most errors came from roses vs. tulips, which share colors and petal textures.

*The dataset is not included.*
