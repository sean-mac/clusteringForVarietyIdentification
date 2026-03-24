

# Consolidated and Enhanced Script for clusteringForVarietyIdentification

import pandas as pd
import numpy as np
import json
import umap
import os
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import itertools
from scipy.spatial.distance import correlation, euclidean
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

# --- Data Formatting (from formatData.py) ---

def processCounts(inFile, outFile):
    df = pd.read_csv(inFile, skiprows=7)
    notData = ['AlleleSequence','SNP','CallRate','OneRatioRef','OneRatioSnp','FreqHomRef','FreqHomSnp','FreqHets','PICRef','PICSnp','AvgPIC','AvgCountRef','AvgCountSnp','RatioAvgCountRefAvgCountSnp']
    df = df.drop(columns=notData)
    df.to_csv(outFile, index=False)

# --- Core Analysis Logic (from base.py, with enhancements) ---

def filterData(countsFile, metaFile, minloci, minSample, refFilter=None, imputation_method='knn', n_neighbors=5):
    print(f"Filtering data and imputing with '{imputation_method}' method...")
    counts = pd.read_csv(countsFile, index_col='MarkerName')
    snpProportion = counts.groupby(['MarkerName']).first() / counts.groupby(['MarkerName']).sum()

    sampleMeta = pd.read_csv(metaFile)
    refRemove = sampleMeta[sampleMeta['reference'] == 'REMOVE']['short_name'].values.astype('str')

    if refFilter:
        divergent = snpProportion.columns[homozygousDivergence(snpProportion) > refFilter].astype('int')
        references = sampleMeta[(sampleMeta['reference'].notna())]['short_name'].values
        refRemove = np.append(refRemove, sampleMeta[sampleMeta['short_name'].isin(np.intersect1d(divergent, references))]['short_name'].values.astype('str'))

    snpProportion = snpProportion.drop(refRemove, axis=1, errors='ignore')
    sampleMeta = sampleMeta.drop(sampleMeta[sampleMeta['short_name'].isin(refRemove.astype('int'))].index)

    snpProportion = snpProportion[snpProportion.isna().sum(axis=1) < (minloci * snpProportion.shape[1])]
    snpProportion = snpProportion.drop(columns=snpProportion.columns[snpProportion.isna().sum(axis=0) > (minSample * snpProportion.shape[0])])
    snpProportionNoInterpolation = snpProportion.copy()

    if imputation_method == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
        snpProportion_imputed = imputer.fit_transform(snpProportion.T).T
        snpProportion = pd.DataFrame(snpProportion_imputed, index=snpProportion.index, columns=snpProportion.columns)
    else: # Fallback to mean imputation
        snpProportion = snpProportion.where(pd.notna(snpProportion), snpProportion.mean(axis=1), axis='rows')

    return snpProportion, snpProportionNoInterpolation, sampleMeta

def embedData(snpProportion, umapSeed):
    print("Embedding data with UMAP...")
    reducer = umap.UMAP(random_state=umapSeed)
    embedding = reducer.fit_transform(snpProportion.T)
    return embedding

def clusteringDBSCAN(embedding, epsilon):
    print(f"Clustering with DBSCAN (epsilon={epsilon})...")
    db = DBSCAN(eps=epsilon, min_samples=1).fit(embedding)
    return db.labels_, None # No probabilities for DBSCAN

def clusteringGMM(embedding, n_clusters):
    print(f"Clustering with Gaussian Mixture Model (n_clusters={n_clusters})...")
    gmm = GaussianMixture(n_components=n_clusters, random_state=0).fit(embedding)
    labels = gmm.predict(embedding)
    probabilities = gmm.predict_proba(embedding)
    return labels, probabilities

def labelSamples(snpProportion, sampleMeta, communities, embedding, cutHeight, admixedCutoff, filePrefix, snpProportionNoInterpolation, parameterFile, probabilities=None):
    print("Assigning variety labels to samples...")
    output = pd.DataFrame(embedding, columns=['embedding_X', 'embedding_Y'])
    output['cluster'] = communities
    output['short_name'] = snpProportion.columns
    output['variety'] = pd.NA
    output['assignment_confidence'] = np.nan

    if probabilities is not None:
        output['assignment_probability'] = np.max(probabilities, axis=1)

    for cluster in np.unique(communities):
        subsetIndex = np.where(communities == cluster)[0]
        cluster_samples = snpProportion.iloc[:, subsetIndex]
        
        # Identify references within the cluster
        references_in_cluster = sampleMeta[sampleMeta['short_name'].isin(cluster_samples.columns.astype(int)) & sampleMeta['reference'].notna()]
        
        if not references_in_cluster.empty:
            # --- Enhanced Assignment Logic ---
            if len(references_in_cluster['reference'].unique()) > 1:
                # Multiple unique references in the cluster, assign to closest
                for sample_idx in subsetIndex:
                    sample_col_name = snpProportion.columns[sample_idx]
                    if int(sample_col_name) in references_in_cluster['short_name'].values:
                        # This sample is a reference itself
                        output.loc[sample_idx, 'variety'] = sampleMeta[sampleMeta['short_name'] == int(sample_col_name)]['reference'].iloc[0]
                        output.loc[sample_idx, 'assignment_confidence'] = 1.0
                    else:
                        # Field sample, find closest reference in this cluster
                        sample_data = snpProportion.iloc[:, sample_idx]
                        closest_ref_name = ''
                        min_dist = np.inf
                        for _, ref_row in references_in_cluster.iterrows():
                            ref_data = snpProportion[str(ref_row['short_name'])]
                            dist = euclidean(sample_data, ref_data)
                            if dist < min_dist:
                                min_dist = dist
                                closest_ref_name = ref_row['reference']
                        output.loc[sample_idx, 'variety'] = closest_ref_name
                        output.loc[sample_idx, 'assignment_confidence'] = 1 - min_dist # Higher is better
            else:
                # Only one unique reference in the cluster, assign all to it
                assigned_variety = references_in_cluster['reference'].unique()[0]
                output.loc[subsetIndex, 'variety'] = assigned_variety
        else:
            # No references, label as a novel genetic entity
            output.loc[subsetIndex, 'variety'] = f'Genetic entity {cluster}'

    # --- Enhanced QA/QC Metrics ---
    het_loci_count = ((snpProportion > 0.05) & (snpProportion < 0.95)).sum(axis=0)
    avg_het_level = snpProportion[((snpProportion > 0.05) & (snpProportion < 0.95))].mean(axis=0)
    output['heterozygous_loci_count'] = het_loci_count.values
    output['avg_heterozygosity_level'] = avg_het_level.reindex(output['short_name']).fillna(0).values
    output['missingness'] = (snpProportionNoInterpolation.isna().sum(axis=0) / snpProportionNoInterpolation.shape[0]).values

    output.to_csv(f"{filePrefix}_clusteringOutput_cutHeight{cutHeight}.csv", index=False)
    return output

# --- Graphing and Helper functions (from graphs.py and others) ---

def homozygousDivergence(x):
    x_vals = x.values if isinstance(x, pd.DataFrame) else x
    _, total = np.unique(np.where((x_vals < 0.2) | (x_vals > 0.8))[1], return_counts=True)
    highDivergence = np.nansum(1 - np.where(x_vals > 0.8, x_vals, np.nan), axis=0)
    lowDivergence = np.nansum(np.where(x_vals < 0.2, x_vals, np.nan), axis=0)
    return (lowDivergence + highDivergence) / total

def umapCluster(embedding, communities):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [6, 0.2]})
    tab20 = mpl.colormaps['tab20'].resampled(max(communities) + 1)
    sc = ax1.scatter(embedding[:, 0], embedding[:, 1], s=5, c=communities, cmap=tab20)
    for i in np.unique(communities):
        samples = np.where(communities == i)
        ax1.text(np.mean(embedding[samples, 0]), np.mean(embedding[samples, 1]), str(i), fontsize=12, weight='bold')
    cbar = plt.colorbar(sc, cax=ax2)
    plt.tight_layout()

# --- Main Execution Block ---

def run_pipeline(parameter_file):
    with open(parameter_file) as f:
        params = json.load(f)

    # Make file paths absolute from the tutorial directory
    dir_path = os.path.dirname(parameter_file)
    inputCountsFile = os.path.join(dir_path, params["inputCountsFile"])
    inputMetaFile = os.path.join(dir_path, params["inputMetaFile"])
    filePrefix = os.path.join(dir_path, params["filePrefix"])

    # --- 1. Data Filtering and Imputation ---
    snpProportion, snpProportionNoInterpolation, sampleMeta = filterData(
        inputCountsFile, 
        inputMetaFile, 
        params["minloci"], 
        params["minSample"],
        imputation_method=params.get("imputation_method", "knn"), # Default to knn
        n_neighbors=params.get("n_neighbors", 5)
    )

    # --- 2. UMAP Embedding ---
    embedding = embedData(snpProportion, params["umapSeed"])

    # --- 3. Clustering ---
    clustering_method = params.get("clustering_method", "dbscan")
    probabilities = None
    if clustering_method == 'gmm':
        # For GMM, we need to estimate or provide the number of clusters
        # Here, we'll use the number of unique references as an estimate
        n_clusters = sampleMeta['reference'].nunique()
        communities, probabilities = clusteringGMM(embedding, n_clusters)
    else: # Default to dbscan
        communities, _ = clusteringDBSCAN(embedding, params["epsilon"])

    # --- 4. Labeling and QA/QC ---
    output_df = labelSamples(
        snpProportion, 
        sampleMeta, 
        communities, 
        embedding, 
        params["cutHeight"], 
        params["admixedCutoff"], 
        filePrefix, 
        snpProportionNoInterpolation, 
        parameter_file,
        probabilities
    )

    # --- 5. Visualization ---
    print("Generating visualizations...")
    umapCluster(embedding, communities)
    plt.savefig(f"{filePrefix}_UMAP_clusters.png", dpi=300)
    plt.close()

    print("\n--- Analysis Complete ---")
    print(f"Output data saved to: {filePrefix}_clusteringOutput_cutHeight{params['cutHeight']}.csv")
    return output_df

if __name__ == '__main__':
    # Set the parameter file path for the tutorial
    parameter_file = '/Users/sean/Desktop/clusteringForVarietyIdentification/tutorial/parametersRiceTutorial.json'
    
    # To use the new features, you can add them to your JSON file, for example:
    # "imputation_method": "knn",
    # "n_neighbors": 5,
    # "clustering_method": "gmm"

    final_output = run_pipeline(parameter_file)
    print("\n--- Final Output Head ---")
    print(final_output.head())

