# WaterMgmt
Water currents clustering

How to run:

# # 2 clusters:

python clusterTrajectoriesDenLAC.py -nclusters 2 -nbins 2 -expFactor 1

Evaluation:

Davies Bouldin Index 0.4600153628534094
Calinski Harabasz Index 28.84254506203523
Silhouette Index 0.226434706584914

# # 3 clusters

python clusterTrajectoriesDenLAC.py -nclusters 3 -nbins 2 -expFactor 1

Evaluation:

Davies Bouldin Index 0.5377738276902567
Calinski Harabasz Index 28.47559219770267
Silhouette Index 0.293641271591822