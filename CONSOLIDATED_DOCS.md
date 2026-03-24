# Consolidated Analysis Scripts

This repository contains three consolidated scripts (`consolidated_analysis_1.py`, `consolidated_analysis_2.py`, and `consolidated_analysis_3.py`) that simplify and enhance the original IDM `clusteringForVarietyIdentification` pipeline. 

Instead of jumping between `base.py`, `formatData.py`, `graphs.py`, and `referenceProcessing.py`, these scripts provide a single-file pipeline for processing DArT tag counts, running dimensionality reduction (UMAP), clustering, and generating visualizations.

## Script Progression

The scripts represent three progressive stages of development, introducing new data science techniques for dealing with sparse genotyping data and classifying varieties.

### `consolidated_analysis_1.py` - The Baseline
This is a direct consolidation of the original IDM codebase into a single, cohesive script. 
* **Core Functionality:** Cleans and formats raw DArT sequences, filters loci and samples based on missingness, and normalizes allele proportions.
* **Dimensionality Reduction:** Uses `UMAP` to project the high-dimensional SNP data down into 2D or 3D space.
* **Clustering:** Uses `DBSCAN` (Density-Based Spatial Clustering of Applications with Noise) to identify dense clusters of technical replicates and reference varieties.
* **Visualizations:** Generates UMAP scatter plots and dendrograms (hierarchical clustering) to validate the distances between technical replicates.

### `consolidated_analysis_2.py` - Imputation and GMMs
This script introduces enhancements for handling missing data and alternative clustering approaches.
* **KNN Imputation:** Introduces `sklearn.impute.KNNImputer`. Instead of just dropping samples or loci with missing calls, it imputes missing SNP proportions based on the $k$-nearest neighbors, recovering more data for analysis.
* **Gaussian Mixture Models (GMM):** Imports `sklearn.mixture.GaussianMixture` to offer probabilistic clustering as an alternative to DBSCAN, allowing for soft-clustering where a sample might have a probability of belonging to multiple overlapping variety clusters.

### `consolidated_analysis_3.py` - Random Forest Classification
The final iteration builds upon the imputed, clustered data to train predictive models.
* **Random Forest Classifier:** Integrates `sklearn.ensemble.RandomForestClassifier`. Once the reference varieties are clustered and labeled in the UMAP space, a Random Forest model is trained to predict the variety of unknown samples.
* **Pairwise Distances:** Uses `pairwise_distances` to better quantify the exact Euclidean/correlation distances between predicted samples and the core reference centroids, allowing for a confidence threshold to be set (e.g., identifying "off-types" that don't match any known variety).

## Usage
To run the analysis, you can simply execute the desired script and call the core functions (like `processCounts()` and `filterData()`) directly within your pipeline. 

```python
import consolidated_analysis_3 as ca

# Process raw counts
ca.processCounts('raw_counts.csv', 'formatted_counts.csv')

# Filter and impute
data = ca.filterData('formatted_counts.csv', 'metadata.csv', minloci=0.8, minSample=0.8, imputation_method='knn')
```
