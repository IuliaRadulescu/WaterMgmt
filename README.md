# WaterMgmt
Water currents clustering

# How to run:

python clusterTrajectoriesDenLAC.py -nclusters [required clusters] -nbins [density bins, between 3 or 4] -expFactor [expansion factor, usually 1]

- See DenLAC algorithm on how to tune params

# Proposed representation method

python clusterTrajectoriesDenLAC.py -nclusters 2 -nbins 2 -expFactor 1

### 2 clusters

Davies Bouldin Index 0.5611420736163114
Calinski Harabasz Index 21.683529358778035
Silhouette Index 0.5987346523736155

### 1 cluster with just one trajectory, the rest of the trajectories in another cluster

Davies Bouldin Index 2.6043644484820954
Calinski Harabasz Index 0.36920272894175094
Silhouette Index 0.16601308693074862

### cluster randomly

Davies Bouldin Index 10.082087470333361
Calinski Harabasz Index 2.4555982310972206
Silhouette Index 0.05145583421432553

### 3 clusters

python clusterTrajectoriesDenLAC.py -nclusters 3 -nbins 3 -expFactor 0.75

Davies Bouldin Index 5.591679515284991
Calinski Harabasz Index 8.327203610565755
Silhouette Index 0.7987412134579661

### 2 cluster each with just one trajectory, the rest of the trajectories in another cluster

Davies Bouldin Index 3.835096506234148
Calinski Harabasz Index 0.33062393769547227
Silhouette Index 0.18000276687323816

### cluster randomly

Davies Bouldin Index 18.79142799212142
Calinski Harabasz Index 1.1538642424887293
Silhouette Index -0.08785332716043567

# Classic representation method

## 2 clusters:

python clusterTrajectoriesDenLAC.py -nclusters 2 -nbins 2 -expFactor 1

Evaluation:

Davies Bouldin Index 0.47012599417505924
Calinski Harabasz Index 28.211970213215327
Silhouette Index 0.20844504729161462

### 1 cluster with just one trajectory, the rest of the trajectories in another cluster

Davies Bouldin Index 0.2779370417009288
Calinski Harabasz Index 3.4595549252953286
Silhouette Index 0.2165951330549933

### cluster randomly

Davies Bouldin Index 3.0638949017947783
Calinski Harabasz Index 8.186919593073492
Silhouette Index 0.008374932761867526

## 3 clusters

python clusterTrajectoriesDenLAC.py -nclusters 3 -nbins 3 -expFactor 0.75

Evaluation:

Davies Bouldin Index 0.5504714435424306
Calinski Harabasz Index 22.560570456914252
Silhouette Index 0.25109400824309897

### 2 cluster each with just one trajectory, the rest of the trajectories in another cluster

Davies Bouldin Index 0.3754186966450796
Calinski Harabasz Index 3.352618010166056
Silhouette Index 0.1815170698976799

### cluster randomly

Davies Bouldin Index 4.209629607738836
Calinski Harabasz Index 6.6173359183332545
Silhouette Index -0.0765362152302252