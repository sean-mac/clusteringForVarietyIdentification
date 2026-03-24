# Consolidated and Enhanced Script for clusteringForVarietyIdentification

import pandas as pd
import numpy as np
import json
from umap.umap_ import UMAP
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
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
    df = df.drop(columns=notData, errors='ignore')
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
    else:
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
    return db.labels_, None

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
        
        references_in_cluster = sampleMeta[sampleMeta['short_name'].isin(cluster_samples.columns.astype(int)) & sampleMeta['reference'].notna()]
        
        if not references_in_cluster.empty:
            if len(references_in_cluster['reference'].unique()) > 1:
                for sample_idx in subsetIndex:
                    sample_col_name = snpProportion.columns[sample_idx]
                    if int(sample_col_name) in references_in_cluster['short_name'].values:
                        output.loc[sample_idx, 'variety'] = sampleMeta[sampleMeta['short_name'] == int(sample_col_name)]['reference'].iloc[0]
                        output.loc[sample_idx, 'assignment_confidence'] = 1.0
                    else:
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
                        output.loc[sample_idx, 'assignment_confidence'] = 1 - min_dist
            else:
                assigned_variety = references_in_cluster['reference'].unique()[0]
                output.loc[subsetIndex, 'variety'] = assigned_variety
        else:
            output.loc[subsetIndex, 'variety'] = f'Genetic entity {cluster}'

    het_loci_count = ((snpProportion > 0.05) & (snpProportion < 0.95)).sum(axis=0)
    avg_het_level = snpProportion[((snpProportion > 0.05) & (snpProportion < 0.95))].mean(axis=0)
    output['heterozygous_loci_count'] = het_loci_count.values
    output['avg_heterozygosity_level'] = avg_het_level.reindex(output['short_name']).fillna(0).values
    output['missingness'] = (snpProportionNoInterpolation.isna().sum(axis=0) / snpProportionNoInterpolation.shape[0]).values

    output.to_csv(f"{filePrefix}_clusteringOutput_cutHeight{cutHeight}.csv", index=False)
    return output

# --- New: Marker Selection Experiment ---

def run_marker_selection_experiment(snpProportion, sampleMeta, n_discriminant_markers=50):
    print("\n--- Running Marker Selection Experiment ---")
    
    # 1. Identify reference and field samples
    all_references_meta = sampleMeta[sampleMeta['reference'].notna()]
    reference_cols = all_references_meta['short_name'].astype(str).tolist()
    reference_cols = [c for c in reference_cols if c in snpProportion.columns]
    field_cols = [c for c in snpProportion.columns if c not in reference_cols]

    if not reference_cols or not field_cols:
        print("Not enough reference or field samples to conduct the experiment.")
        return

    # 2. Identify discriminant markers using only references
    print(f"Identifying top {n_discriminant_markers} discriminant markers from reference samples...")
    X_ref = snpProportion[reference_cols].T
    y_ref = sampleMeta.set_index('short_name').loc[X_ref.index.astype(int)]['reference']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_ref, y_ref)
    
    feature_importances = pd.Series(rf.feature_importances_, index=snpProportion.index)
    discriminant_markers = feature_importances.nlargest(n_discriminant_markers).index.tolist()
    background_markers = feature_importances.index.difference(discriminant_markers).tolist()

    # 3. Perform forced-choice assignment for each marker set
    def assign_to_closest(field_data, reference_data):
        assignments = {}
        distances = pairwise_distances(field_data.T, reference_data.T, metric='euclidean')
        closest_indices = np.argmin(distances, axis=1)
        assignments = reference_data.columns[closest_indices].tolist()
        # Get the actual variety name from the column name (short_name)
        variety_names = sampleMeta.set_index('short_name').loc[pd.to_numeric(assignments)]['reference'].tolist()
        return variety_names

    print("Assigning field samples based on different marker sets...")
    # Assignment using ALL markers
    assignment_all = assign_to_closest(snpProportion.loc[:, field_cols], snpProportion.loc[:, reference_cols])
    # Assignment using DISCRIMINANT markers
    assignment_disc = assign_to_closest(snpProportion.loc[discriminant_markers, field_cols], snpProportion.loc[discriminant_markers, reference_cols])
    # Assignment using BACKGROUND markers
    assignment_back = assign_to_closest(snpProportion.loc[background_markers, field_cols], snpProportion.loc[background_markers, reference_cols])

    # 4. Create and display the comparison report
    report = pd.DataFrame({
        'field_sample_id': field_cols,
        'assignment_all_markers': assignment_all,
        'assignment_discriminant_only': assignment_disc,
        'assignment_background_only': assignment_back
    })
    
    print("\n--- Marker Selection Experiment Report ---")
    print(report)
    report.to_csv(os.path.join(os.path.dirname(sampleMeta.iloc[0]['filePrefix']), "marker_selection_report.csv"), index=False)
    print("\nMarker selection report saved to marker_selection_report.csv")

# --- Graphing and Helper functions ---

def homozygousDivergence(x):
    x_vals = x.values if isinstance(x, pd.DataFrame) else x
    _, total = np.unique(np.where((x_vals < 0.2) | (x_vals > 0.8))[1], return_counts=True)
    highDivergence = np.nansum(1 - np.where(x_vals > 0.8, x_vals, np.nan), axis=0)
    lowDivergence = np.nansum(np.where(x_vals < 0.2, x_vals, np.nan), axis=0)
    return (lowDivergence + highDivergence) / total if total.size > 0 else np.zeros(x_vals.shape[1])

def umapCluster(embedding, communities):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [6, 0.2]})
    unique_communities = np.unique(communities)
    tab20 = mpl.colormaps['tab20'].resampled(len(unique_communities))
    scatter_colors = [tab20(np.where(unique_communities == c)[0][0]) for c in communities]
    sc = ax1.scatter(embedding[:, 0], embedding[:, 1], s=5, c=scatter_colors)
    for i in unique_communities:
        samples = np.where(communities == i)
        ax1.text(np.mean(embedding[samples, 0]), np.mean(embedding[samples, 1]), str(i), fontsize=12, weight='bold')
    cbar = plt.colorbar(sc, cax=ax2)
    plt.tight_layout()

# --- Main Execution Block ---

def run_pipeline(parameter_file):
    with open(parameter_file) as f:
        params = json.load(f)

    dir_path = os.path.dirname(parameter_file)
    inputCountsFile = os.path.join(dir_path, params["inputCountsFile"])
    inputMetaFile = os.path.join(dir_path, params["inputMetaFile"])
    filePrefix = os.path.join(dir_path, params["filePrefix"])
    sampleMeta = pd.read_csv(inputMetaFile)
    sampleMeta['filePrefix'] = filePrefix # Add for later use

    snpProportion, snpProportionNoInterpolation, sampleMeta = filterData(
        inputCountsFile, inputMetaFile, params["minloci"], params["minSample"],
        imputation_method=params.get("imputation_method", "knn"), n_neighbors=params.get("n_neighbors", 5)
    )

    embedding = embedData(snpProportion, params["umapSeed"])

    clustering_method = params.get("clustering_method", "dbscan")
    probabilities = None
    if clustering_method == 'gmm':
        n_clusters = sampleMeta['reference'].nunique()
        communities, probabilities = clusteringGMM(embedding, n_clusters)
    else:
        communities, _ = clusteringDBSCAN(embedding, params["epsilon"])

    output_df = labelSamples(
        snpProportion, sampleMeta, communities, embedding, params["cutHeight"], 
        params["admixedCutoff"], filePrefix, snpProportionNoInterpolation, parameter_file, probabilities
    )

    print("Generating visualizations...")
    umapCluster(embedding, communities)
    plt.savefig(f"{filePrefix}_UMAP_clusters.png", dpi=300)
    plt.close()

    # --- Run Marker Selection Experiment ---
    if params.get("run_marker_experiment", False):
        run_marker_selection_experiment(snpProportion, sampleMeta, n_discriminant_markers=params.get("n_discriminant_markers", 50))

    print("\n--- Analysis Complete ---")
    print(f"Output data saved to: {filePrefix}_clusteringOutput_cutHeight{params['cutHeight']}.csv")
    return output_df

if __name__ == '__main__':
    parameter_file = '/Users/sean/Desktop/clusteringForVarietyIdentification/tutorial/parametersRiceTutorial.json'
    
    # To run the marker selection experiment, add the following to your JSON file:
    # "run_marker_experiment": true,
    # "n_discriminant_markers": 50

    final_output = run_pipeline(parameter_file)
    print("\n--- Final Output Head ---")
    print(final_output.head())