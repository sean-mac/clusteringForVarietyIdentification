#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check of reference similarity
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.spatial.distance import correlation
import scipy.cluster.hierarchy as sch
import graphs as plot


#Generate the histogram of technical replicate divergence
#pick a reasonable cutoff value --> throw out/manually curate these references
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

def heatmapTechnicalRep(snpProportion, sampleMeta, percentile):
    """
    Generate a heatmap and dendrogram for each cluster

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        divergentDistance: list of divergence values for samples
        divergentInventory: sample names in the same order as divergentDistance
	"""
    distance, sample = histogramTechnicalRep(snpProportion, sampleMeta)
    
    cutoff = distance[np.argsort(distance)[int(np.floor(percentile*len(distance)))]]
    divergentDistance = distance[distance > cutoff]
    divergentInventory = sample[distance > cutoff]

    refSubset = sampleMeta[pd.notna(sampleMeta['reference_original'])]
    w = refSubset[refSubset['short_name'].isin(snpProportion.columns.astype('int'))]

    columnOrder = np.argsort(divergentDistance) #increasing distance
    
    refShort = []
    inventoryLabels = []
    for i in divergentInventory[columnOrder]:
        refShort.append(w[w['inventory'] == i]['short_name'].values)
        inventoryLabels += [i]*len(w[w['inventory'] == i]['short_name'].values)
    refShort = np.concatenate(refShort).ravel().astype('str')
    
    countsSubset = snpProportion[refShort].values
    
    ax1, ax2 = plot.plotTemplate()
    SC=ax1.imshow(countsSubset, aspect='auto', interpolation='none', cmap='coolwarm',vmin=0,vmax = 1)
    ax1.set_yticks([])
    plt.colorbar(SC, cax=ax2)
    ax1.set_xticks(np.arange(len(refShort)), refShort, rotation = 90)
    plt.tight_layout()

def splitReferences(snpProportion, sampleMeta, communities):
    """
    Generate a heatmap and dendrogram for each cluster

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        communities: DBSCAN cluster number for each sample
	"""
    print('Varieties that are split across multiple clusters')
    
    refSubset = sampleMeta[pd.notna(sampleMeta['reference_original'])]
    w = refSubset[refSubset['short_name'].isin(snpProportion.columns.astype('int'))]
    
    splitVar = []
    for ref in np.unique(w['reference_original']):
        shortNames = w[w['reference_original'] == ref]['short_name'].values.astype('str')
        if len(np.unique(communities[np.where(np.isin(snpProportion.columns, shortNames))[0]])) > 1:
            splitVar.append(ref)
            print(ref, np.unique(communities[np.where(np.isin(snpProportion.columns, shortNames))[0]]))
            

def referenceDistance(snpProportion, sampleMeta):
    """
    Generate a scatterplot with the max distance between all replicates of the same reference  

    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
	"""
    refSubset = sampleMeta[pd.notna(sampleMeta['reference'])]
    w = refSubset[refSubset['short_name'].isin(snpProportion.columns.astype('int'))]
    
    maxDistance = []
    for ref in np.unique(w['reference']):
        shortNames = w[w['reference_original'] == ref]['short_name'].values.astype('str')
        if len(shortNames) > 1:
            maxDistance.append(sch.linkage(snpProportion[shortNames].values.T, metric='correlation')[-1,2])
    
    plt.figure()
    plt.plot(np.arange(len(maxDistance)), np.sort(maxDistance), 'bo-')
    plt.ylabel('Max distance')
    plt.xlabel('Proportion of references')
    plt.xticks(len(maxDistance)*np.arange(0,1.1,0.1), np.around(np.arange(0,1.1,0.1),1))
    plt.tight_layout()