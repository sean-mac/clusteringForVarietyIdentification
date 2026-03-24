

# Consolidated script for clusteringForVarietyIdentification
# Combines logic from base.py, formatData.py, graphs.py, and referenceProcessing.py

import pandas as pd
import numpy as np
import json

import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import itertools
from scipy.spatial.distance import correlation
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import umap
from umap import UMAP

# --- From formatData.py ---

def processCounts(inFile, outFile):
    '''
    Process a DArT tag counts file with seven header rows into the correct input format
    
    Args:
        inFile: name and path to counts file
        outFile: name and path for output file
    '''
    df = pd.read_csv(inFile, skiprows=7)
    notData = ['AlleleSequence','SNP','CallRate','OneRatioRef',
            'OneRatioSnp','FreqHomRef','FreqHomSnp','FreqHets',
            'PICRef','PICSnp','AvgPIC','AvgCountRef','AvgCountSnp',
            'RatioAvgCountRefAvgCountSnp']
    df = df.drop(columns=notData)
    df.to_csv(outFile, index=False)    
 
def processCountsSeq(inFile, outFile):  
    '''
    Process a DArT seq counts file with seven header rows into the correct input format
    
    Args:
        inFile: name and path to counts file
        outFile: name and path for output file
    '''  
    df = pd.read_csv(inFile, skiprows =7)
    notData = ['AlleleID', 'AlleleSequence', 'TrimmedSequence',
           'Chrom_Eragrostis_CogeV3', 'ChromPosTag_Eragrostis_CogeV3',
           'ChromPosSnp_Eragrostis_CogeV3', 'AlnCnt_Eragrostis_CogeV3',
           'AlnEvalue_Eragrostis_CogeV3', 'Strand_Eragrostis_CogeV3', 'SNP',
           'SnpPosition', 'CallRate', 'OneRatioRef', 'OneRatioSnp', 'FreqHomRef',
           'FreqHomSnp', 'FreqHets', 'PICRef', 'PICSnp', 'AvgPIC', 'AvgCountRef',
           'AvgCountSnp', 'RepAvg']
    df = df.drop(columns=notData)    
    df.rename(columns={'CloneID': 'MarkerName'}, inplace=True)
    df.to_csv(outFile, index=False)

# --- From graphs.py ---

def clusterReorder(subset, counts):
    """
    Sort genes and samples using heirarchical clustering
    
    Args:
        subset: processed SNP proportion data (values only)
        counts: list of the number of samples per DBSCAN cluster
    """
    #sort genes using heirarchical clustering
    Y_gene = sch.linkage(subset, metric='euclidean')
    Z_gene = sch.leaves_list(Y_gene)
    subsetReorder = subset[Z_gene,:]

    #sort samples within DBSCAN clusters using heirarchical clustering
    breakPoints = [0]+list(np.cumsum(counts))
    clusterOrder = []
    for i, j in enumerate(counts):
        if j > 1:
            clusterSubset = subset[:,breakPoints[i]:breakPoints[i+1]]
            Y_cluster = sch.linkage(clusterSubset.T, metric='correlation')
            Z_cluster = sch.leaves_list(Y_cluster)
            if i != 0: 
                Z_cluster += breakPoints[i]
            clusterOrder += list(Z_cluster)
        else:
            clusterOrder += [breakPoints[i]]         

    return subsetReorder[:,clusterOrder], clusterOrder, breakPoints

def homozygousDivergence(x):
    """
    Calculate the divergence for a numpy array of processed SNP proportion data
    """
    _, total = np.unique(np.where((x < 0.2) | (x > 0.8))[1], return_counts=True)
    highDivergence = np.nansum(1 - np.where(x > 0.8, x, np.nan), axis = 0)
    lowDivergence = np.nansum(np.where(x < 0.2, x, np.nan), axis = 0)
    return (lowDivergence + highDivergence)/ total

def plotTemplate():
    """
    Basic layout for a figure with one plot and a colorbar
    """
    fig = plt.figure(figsize=(7.2,6))
    gs=GridSpec(1,2, width_ratios=[6,0.2], figure = fig)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    return ax1, ax2

def plotDouble():
    """
    Basic layout for a figure with two plots
    """
    fig = plt.figure(figsize=(10,4))
    gs = GridSpec(1, 2, width_ratios=[1,1], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    return ax1, ax2

def umapCluster(embedding, communities):
    """
    UMAP with samples colored by cluster
    
    Args:
        embedding: UMAP embedding of data 
        communities: DBSCAN cluster number for each sample
    """
    ax1, ax2 = plotTemplate()
    tab20 = mpl.colormaps['tab20'].resampled(max(communities)+1)
    SC=ax1.scatter(embedding[:, 0], embedding[:, 1], s = 1, c=communities, cmap = tab20)
    
    #add cluster labels
    for i in np.unique(communities):
        samples = np.where(communities == i)
        ax1.text(np.mean(embedding[samples,0]), np.mean(embedding[samples,1]), i, fontsize=12)
    
    #add colorbar
    cbar = plt.colorbar(SC, cax=ax2)
    boxWidth = max(communities)/(max(communities)+1)
    cbar.set_ticks((np.linspace(0,max(communities),max(communities)+2)[:-1]+boxWidth/2), labels = np.arange(max(communities)+1).astype('str'))
    plt.tight_layout()

def umapReference(snpProportion, embedding, sampleMeta, communities):
    """
    UMAP with references colored by cluster 
    
    Args:
        snpProportion: processed SNP proportion data
        embedding: UMAP embedding of data 
        sampleMeta: metadata paired with genotyping data
        communities: DBSCAN cluster number for each sample
    """
    references = sampleMeta[(sampleMeta['reference'].notna())]
    referencesIndex = np.where(np.isin(snpProportion.columns, references['short_name'].astype('str')))[0]
    refClusters, refCounts = np.unique(communities[referencesIndex], return_counts=True)
    
    ax1, ax2 = plotTemplate()
    tab20 = mpl.colormaps['tab20'].resampled(len(refClusters))
    ax1.scatter(embedding[:, 0], embedding[:, 1], s = 1, c='grey', alpha = 0.1)
    labels = []
    bot = 0
    
    for i, j in enumerate(refClusters): #all the ref in a cluster should be the same color
        index = np.intersect1d(np.where(communities == j), referencesIndex)
        ax1.scatter(embedding[index, 0], embedding[index, 1], s = 3, color=tab20(i))
        
        #add cluster labels
        samples = np.where(communities == j)
        ax1.text(np.mean(embedding[samples,0]), np.mean(embedding[samples,1]), j, fontsize=12)
    
        #add legend
        shortName = snpProportion.columns[index].astype('int')
        variety = sampleMeta[sampleMeta['short_name'].isin(shortName)]['reference']    
        labels += np.unique(variety).tolist()
        ax2.bar(0,len(np.unique(variety)),bottom = bot, color = tab20(i)) 
        bot += len(np.unique(variety))  
            
    ax2.yaxis.tick_right()    
    ax2.set_yticks(np.arange(len(labels))+0.5,labels, fontsize=5) #need to offset labels by box width
    ax2.get_xaxis().set_visible(False)
    ax2.set_ylim((0,bot))
    plt.tight_layout()

def barchartRef(snpProportion, output, sampleMeta):
    """
    Barchart with the prevalence of each observed reference variety

    Args:
        snpProportion: processed SNP proportion data
        output: output dataframe frome base.py
        sampleMeta: metadata paired with genotyping data
    """
    w = sampleMeta[sampleMeta['short_name'].isin(snpProportion.columns.astype('int'))]
    var, counts = np.unique(output['variety'].dropna(), return_counts=True)
    
    refshort = w['short_name'][(sampleMeta['reference'].notna())].values.astype('str') #references
    admixedshort = output['short_name'][output['variety'] == 'Admixed'].values.astype('str') #admixed
    
    landraceName = var[np.flatnonzero(np.core.defchararray.find(var.astype('str'),'Genetic entity')!=-1)]
    landraceShort = output['short_name'][np.isin(output['variety'],landraceName)].values.astype('str')
    
    callshort = np.setdiff1d(output['short_name'],np.concatenate((refshort, admixedshort,landraceShort)))
    callVarieties, callVarietiesCount = np.unique(output[output['short_name'].isin(callshort)]['variety'].dropna(), return_counts=True)

    fig, ax = plt.subplots(figsize=(14.4,4.8))
    ax.barh(np.arange(len(callVarieties)), callVarietiesCount[np.argsort(callVarietiesCount)])
    ax.bar_label(ax.containers[0], label_type='edge')
    ax.set_yticks(np.arange(len(callVarieties)),callVarieties[np.argsort(callVarietiesCount)])
    plt.tight_layout()

# --- From base.py ---

def filterData(countsFile, metaFile, minloci, minSample, refFilter = None):
    '''
    Input the reformatted counts file and paired metadata file, filter out low quality samples/genes and then interpolate missing data 
    
    Args:
        countsFile: path to reformatted counts file
        metaFile: path to the metadata file paired with the countsFile
        minSample: samples must be missing counts data from less than X proportion of markers
        minloci: markers must be have counts data from more than than Y proportion of samples
        refFilter: (optional) remove references with a divergence score above this value
    '''
    #import counts data
    counts = pd.read_csv(countsFile, index_col='MarkerName')
    snpProportion = counts.groupby(['MarkerName']).first()/counts.groupby(['MarkerName']).sum() #if the count for both SNPs are zero --> NaN
    
    #check that all marker names are unique
    marker, markerCount = np.unique(counts.index, return_counts=True)
    if len(np.where(markerCount > 2)[0]) > 0: print('More than two rows with the same marker name, please differentiate marker names in the count file')
    
    #import sample metadata
    sampleMeta = pd.read_csv(metaFile)
    refRemove = sampleMeta[sampleMeta['reference'] == 'REMOVE']['short_name'].values.astype('str')
    
    #optionally remove references above a divergence cutoff
    if refFilter:
        divergent = snpProportion.columns[homozygousDivergence(snpProportion) > refFilter].astype('int') #all samples above cutoff
        references = sampleMeta[(sampleMeta['reference'].notna())]['short_name'].values
        refRemove = np.append(refRemove, sampleMeta[sampleMeta['short_name'].isin(np.intersect1d(divergent, references))]['short_name'].values.astype('str'))
        
    snpProportion = snpProportion.drop(refRemove, axis=1)
    sampleMeta = sampleMeta.drop(sampleMeta[sampleMeta['short_name'].isin(refRemove.astype('int'))].index)
    
    snpProportionNoInterpolation = snpProportion.copy()
    
    #filter snpProportion for samples and genes with too many NaN
    snpProportion = snpProportion[snpProportion.isna().sum(axis = 1) < (minloci*snpProportion.shape[1])] #remove genes
    snpProportion = snpProportion.drop(columns=snpProportion.columns[snpProportion.isna().sum(axis = 0) > (minSample*snpProportion.shape[0])]) #remove samples

    #interpolate NaN using gene average
    snpProportion = snpProportion.where(pd.notna(snpProportion), snpProportion.mean(axis = 1), axis='rows')
    
    return snpProportion, snpProportionNoInterpolation, sampleMeta

def embedData(snpProportion, umapSeed):
    '''
    Input the processed snpProprtion data, embed with UMAP, and then cluster using DBSCAN
    
    Args:
        snpProportion: processed SNP proportion data
        umapSeed: RNG seed for umap embedding
    '''
    #UMAP embedding
    reducer = umap.UMAP(random_state=umapSeed)
    embedding = reducer.fit_transform(snpProportion.T) #the order is the same after embedding
    
    return embedding

def clusteringDBSCAN(snpProportion, sampleMeta, embedding, epsilon, filePrefix, admixedCutoff):
    '''
    Input the processed snpProprtion data, embed with UMAP, and then cluster using DBSCAN
    
    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        embedding: UMAP embedding of snpProportion
        epsilon: epsilon parameter for DBSCAN clustering
        filePrefix: prefix for output filenames
    '''    
    #cluster using DBSCAN
    db_communities = DBSCAN(eps=epsilon, min_samples=1).fit(embedding).labels_
    
    #save output figures
    umapCluster(embedding, db_communities)
    plt.savefig(filePrefix+' UMAP DBSCAN (epsilon ' + str(epsilon)+').png', dpi = 300)
    
    umapReference(snpProportion, embedding, sampleMeta, db_communities)
    plt.savefig(filePrefix+' UMAP references (DBSCAN clusters, epsilon ' + str(epsilon)+').png', dpi = 300)

    if admixedCutoff:
        histogramDivergence(snpProportion,sampleMeta)
        plt.savefig(filePrefix+' histogram divergence.png', dpi = 300)
   
    return db_communities

def labelSamples(snpProportion,sampleMeta,db_communities,embedding, cutHeight, admixedCutoff, filePrefix, snpProportionNoInterpolation, parameterFile):
    '''
    Evaluate different cut height values for processing the dendrogram
    
    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        db_communities: DBSCAN cluster number for each sample
        embedding: UMAP embedding of snpProportion
        cutHeight: cutoff value for cutting a dendrogram into clusters
        admixedCutoff: clades without a reference and a minimum divergence value above this will be labeled as admixed
        filePrefix: prefix for output filenames
    '''
    #consolidate outputs
    output = pd.DataFrame(embedding, columns=['embedding_X', 'embedding_Y'])
    output['cluster'] = db_communities
    output['short_name'] = snpProportion.columns
    if admixedCutoff:
        output['divergence'] = homozygousDivergence(snpProportion)
    output['variety'] = pd.NA    
    
    for cluster in np.unique(db_communities):
        #subset for a single DBSCAN cluster
        subsetIndex = np.where(db_communities == cluster)[0]
            
        #cluster subset of samples using heirarchical clustering
        Y_cluster = sch.linkage(snpProportion[snpProportion.columns[subsetIndex]].values.T, metric='correlation')
        
        #label samples
        # Note: rand.labelHCLandrace is not defined in the provided scripts. This will cause an error.
        # I will comment it out for now.
        # communities, names = rand.labelHCLandrace(snpProportion[snpProportion.columns[subsetIndex]], sampleMeta, Y_cluster, cutHeight, clusterNumber = cluster, admixedCutoff = admixedCutoff)
        # varietiesList = []
        # for i in communities.astype('int'):
        #     varietiesList.append(names[i][0])          
        # output.loc[subsetIndex,'variety'] = varietiesList
        pass # Placeholder
    
    #save outputs
    # Note: umapRefLandrace is not defined in the provided scripts. This will cause an error.
    # I will comment it out for now.
    # umapRefLandrace(snpProportion, output, sampleMeta, 5, noRef=True)
    # plt.savefig(filePrefix+' UMAP clustering predictions (cut height'+str(cutHeight)+').png', dpi = 300)
    
    barchartRef(snpProportion, output, sampleMeta)
    plt.savefig(filePrefix+' bar chart clustering predictions (cut height'+str(cutHeight)+').png', dpi = 300)
        
    #add missingness
    output['missingness'] = ((snpProportionNoInterpolation.isna().sum(axis = 0)[snpProportion.columns])/snpProportionNoInterpolation.shape[1]).values
    
    #add heterozygosity
    output['heterozygosity'] = (((snpProportion > 0.05) & (snpProportion < 0.95)).sum(axis=0) / snpProportion.shape[0]).values
    
    output.to_csv(filePrefix+'_clusteringOutputData_cutHeight'+str(cutHeight)+'.csv', index=False)
    
    #append sampleMeta
    metaOrder = []
    for i in output['short_name']:
         metaOrder.append(np.where(sampleMeta['short_name'] == int(i))[0][0])
     
    sampleMetaCrop = sampleMeta.iloc[metaOrder]
    sampleMetaCrop.index = output.index
    output2 = pd.concat([output, sampleMetaCrop], axis=1)
     
    #add paramters
    with open(parameterFile) as f:
        data = json.load(f)
        
    output2['parameters'] = np.nan    
    output2['parameters'].iloc[0:len(data)] = list(dict.items(data))
    output2.to_csv(filePrefix+'_clusteringOutputAllData_cutHeight'+str(cutHeight)+'.csv', index=False)
     
    return output, output2

def loadParameters(parameterFile):
    '''
    load a json file with all of the parameters
    '''
    with open(parameterFile) as f:
        data = json.load(f)
        
    minSample = data["minSample"]
    minloci = data["minloci"]
    umapSeed = data["umapSeed"]
    epsilon = data["epsilon"]
    cutHeight = data["cutHeight"]
    admixedCutoff = data["admixedCutoff"]
    filePrefix = data["filePrefix"]
    inputCountsFile = data["inputCountsFile"]
    inputMetaFile = data["inputMetaFile"]
    
    return minSample, minloci, umapSeed, epsilon, cutHeight, admixedCutoff, filePrefix, inputCountsFile, inputMetaFile

# --- From referenceProcessing.py ---

def histogramTechnicalRep(snpProportion, sampleMeta):
    """
    Generate a heatmap and dendrogram for each cluster

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
	"""
    refSubset = sampleMeta[pd.notna(sampleMeta['reference_original'])]
    w = refSubset[refSubset['short_name'].isin(snpProportion.columns.astype('int'))]
    repNames, repCounts = np.unique(w['inventory'], return_counts=True)
    distance = []
    sample = []
    for i in repNames[repCounts > 1]:
        shortName = w[w['inventory'] == i]['short_name'].values.astype('str')
        subset = snpProportion[snpProportion.columns[np.isin(snpProportion.columns,shortName)]].values
    
        for j in list(itertools.combinations(np.arange(len(shortName)), 2)): #if there are more than two technical rep, do all pairwise combinations
            distance.append(correlation(subset[:,j[0]],subset[:,j[1]]))  
            sample.append(i)
            
    
    #histogram of distance between technical replicates
    plt.figure()
    plt.hist(distance, bins = 20)
    plt.xlabel('Techincal replicate distance (correlation)')
    plt.tight_layout()

    return np.asarray(distance), np.asarray(sample)

def runReferenceProcessing(parameterFile):
    minSample, minloci, umapSeed, epsilon, cutHeight, admixedCutoff, filePrefix, inputCountsFile, inputMetaFile = loadParameters(parameterFile)
    
    # Make file paths absolute
    dir_path = os.path.dirname(parameterFile)
    inputCountsFile = os.path.join(dir_path, inputCountsFile)
    inputMetaFile = os.path.join(dir_path, inputMetaFile)
    filePrefix = os.path.join(dir_path, filePrefix)

    snpProportion, snpProportionNoInterpolation, sampleMeta = filterData(inputCountsFile, inputMetaFile, minloci, minSample)
    embedding = embedData(snpProportion, umapSeed)
    db_communities = clusteringDBSCAN(snpProportion, sampleMeta, embedding, epsilon, filePrefix, admixedCutoff)
    
    distance, sample = histogramTechnicalRep(snpProportion, sampleMeta)
    plt.savefig(filePrefix+' histogram technical replicate distance.png', dpi = 300)
    
    # Note: heatmapTechnicalRep is not defined in the provided scripts. This will cause an error.
    # I will comment it out for now.
    # heatmapTechnicalRep(snpProportion, sampleMeta, 0.9)
    # plt.savefig(filePrefix+' heatmap technical replicate distance.png', dpi = 300)
    
    # Note: splitReferences is not defined in the provided scripts. This will cause an error.
    # I will comment it out for now.
    # splitReferences(snpProportion, sampleMeta, db_communities)
    
    # Note: referenceDistance is not defined in the provided scripts. This will cause an error.
    # I will comment it out for now.
    # referenceDistance(snpProportion, sampleMeta)
    # plt.savefig(filePrefix+' reference distance.png', dpi = 300)

# --- Main Execution Block ---

if __name__ == '__main__':
    # Set the parameter file path
    parameterFile = '/Users/sean/Desktop/clusteringForVarietyIdentification/tutorial/parametersRiceTutorial.json'
    
    print("--- Running Main Analysis Pipeline ---")
    minSample, minloci, umapSeed, epsilon, cutHeight, admixedCutoff, filePrefix, inputCountsFile, inputMetaFile = loadParameters(parameterFile)
    
    # Make file paths absolute
    dir_path = os.path.dirname(parameterFile)
    inputCountsFile = os.path.join(dir_path, inputCountsFile)
    inputMetaFile = os.path.join(dir_path, inputMetaFile)
    filePrefix = os.path.join(dir_path, filePrefix)

    snpProportion, snpProportionNoInterpolation, sampleMeta = filterData(inputCountsFile, inputMetaFile, minloci, minSample)
    embedding = embedData(snpProportion, umapSeed)
    db_communities = clusteringDBSCAN(snpProportion, sampleMeta, embedding, epsilon, filePrefix, admixedCutoff)
    output, output2 = labelSamples(snpProportion, sampleMeta, db_communities, embedding, cutHeight, admixedCutoff, filePrefix, snpProportionNoInterpolation, parameterFile)

    print("\n--- Running Reference Processing ---")
    runReferenceProcessing(parameterFile)

    print("\n--- Analysis Complete ---")

