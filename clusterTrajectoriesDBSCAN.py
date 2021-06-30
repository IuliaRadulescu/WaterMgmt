import numpy as np
import pandas as pd
import utils
import similaritymeasures
from sklearn.cluster import DBSCAN
import plotTrajectoriesHYSPLIT

trajDf = utils.readTraj()
groupedByTraj = trajDf.groupby('ntra')

trajectoryDict = {}

for ntra, group in groupedByTraj:
    trajectoryDict[ntra-1] = np.array(group[['lat', 'lon']])

# compute distance matrix
dictLen = len(trajectoryDict)
distMat = np.zeros((dictLen, dictLen), dtype=np.float64)

for ntra1 in range(dictLen):
    for ntra2 in range(dictLen):
        distMat[ntra1][ntra2] = similaritymeasures.frechet_dist(trajectoryDict[ntra1], trajectoryDict[ntra2])

clustering = DBSCAN(eps=3, min_samples=5, metric='precomputed').fit(distMat)

print(clustering.labels_)

for ntra, group in groupedByTraj:
    group['labelDBSCAN'] = [clustering.labels_[ntra-1]] * group['label'].shape[0]
    if ntra == 1:
        resultDf = group
    else:
        resultDf = pd.concat([resultDf, group], ignore_index=True, sort=False)

plotTrajectoriesHYSPLIT.plotTraj(resultDf, 'labelDBSCAN')



    