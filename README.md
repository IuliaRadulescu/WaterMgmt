# WaterMgmt
Water currents clustering

How to run:

# # 2 clusters:

python clusterTrajectoriesDenLAC.py -nclusters 2 -nbins 2 -expFactor 1

Evaluation:

Davies Bouldin Index 0.47012599417505924
Calinski Harabasz Index 28.211970213215327
Silhouette Index 0.20844504729161462

# # 3 clusters

python clusterTrajectoriesDenLAC.py -nclusters 3 -nbins 3 -expFactor 0.75

Evaluation:

Davies Bouldin Index 0.5504714435424306
Calinski Harabasz Index 22.560570456914252
Silhouette Index 0.25109400824309897