import numpy as np
import pandas as pd
import os
import math
from math import radians, degrees
import itertools
from sklearn.metrics.pairwise import haversine_distances
import matplotlib
import matplotlib.pyplot as plt
import random
from astropy.coordinates import SphericalRepresentation

import utils
import plotTrajectoriesHYSPLIT
import denlac

def convertToCartesian(elem):

    R = 1

    lon_r = elem[0]
    lat_r = elem[1]

    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    
    return (x, y)

def generateDenLACCoords(distance_type):

    trajDf = utils.readTraj()

    trajDf['lat_r'] = trajDf.lat.apply(radians)
    trajDf['lon_r'] = trajDf.lon.apply(radians)

    groupedByTraj = trajDf.groupby('ntra')

    trajectoryDict = {}

    for ntra, group in groupedByTraj:
        latLonArray = np.array(group[['lat_r', 'lon_r']])
        trajectoryDict[ntra-1] = np.array([convertToCartesian(elem) for elem in latLonArray])

    trajectoryLens = []

    for key, elem in trajectoryDict.items():
        trajectoryLens.append(np.shape(elem)[0])

    maxLen = max(trajectoryLens)

    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    fileLocation = scriptDirectory + '/trajectories/czech_june_2021/'
    trajDenLACFile = open(fileLocation + 'trajsDenLAC_' + distance_type + '.txt', 'w')

    dataset = []

    for key, elem in trajectoryDict.items():

        if (len(elem) < maxLen):
            continue

        listToAppend = elem.tolist()
        listToAppend.extend([key])
        dataset.append(listToAppend)

        line = ','.join(map(str, elem))
        line += ',' + str(key) + '\n'
        trajDenLACFile.write(line)

    trajDenLACFile.close()

    return dataset


def getClustersForDatasetElements(datasetWithLabels, clusterPoints):
        
    point2pointId = {}
    point2clusterId = {}

    for point in datasetWithLabels:
        index = np.array([point[0:-1]])
        point2pointId[tuple(index.flatten().tolist())] = point[-1]

    for clusterId, elementsInCluster in clusterPoints.items():
        for element in elementsInCluster:
            point2clusterId[tuple(element.flatten().tolist())] = clusterId

    return [(point2pointId[point], point2clusterId[point]) if point in point2clusterId.keys() else (point2pointId[point], -1) for point in point2pointId.keys()]

def plotDenLACResult(denLACResult):

    trajDf = utils.readTraj()

    groupedByTraj = trajDf.groupby('ntra')

    for ntra, group in groupedByTraj:
        if (ntra-1) not in denLACResult:
            continue
        group['labelDenLAC'] = [denLACResult[ntra-1]] * group['label'].shape[0]
        if ntra == 1:
            resultDf = group
        else:
            resultDf = pd.concat([resultDf, group], ignore_index=True, sort=False)

    plotTrajectoriesHYSPLIT.plotTraj(resultDf, 'labelDenLAC')

def plotPlaneProjection(denLACResult):

    trajDf = utils.readTraj()

    trajDf['lat_r'] = trajDf.lat.apply(radians)
    trajDf['lon_r'] = trajDf.lon.apply(radians)

    groupedByTraj = trajDf.groupby('ntra')

    trajectoryDict = {}

    for ntra, group in groupedByTraj:
        latLonArray = np.array(group[['lat_r', 'lon_r']])
        trajectoryDict[ntra-1] = np.array([convertToCartesian(elem) for elem in latLonArray])
        
    nrColors = len(set(denLACResult.values()))

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(nrColors)]

    usedColors = {}

    for trajId in trajectoryDict:
        if trajId not in denLACResult:
            continue
        if denLACResult[trajId] not in usedColors:
            usedColors[denLACResult[trajId]] = colors[denLACResult[trajId]]
        traj = trajectoryDict[trajId]
        plt.plot(traj[:, 0], traj[:, 1], label = 'traj ' + str(trajId), color = colors[denLACResult[trajId]])

    print('COLORS = ', usedColors)
    
    plt.show()


datasetWithLabels = generateDenLACCoords('euclidean')

dataset = np.array([elem[0:-1] for elem in datasetWithLabels])

print('Dataset of shape', np.shape(dataset))

joinedPartitions = denlac.runDenLAC(dataset)

# print('OUTPUT ==', joinedPartitions)

points2ClustersDict = dict(getClustersForDatasetElements(datasetWithLabels, joinedPartitions))

print(points2ClustersDict)

plotPlaneProjection(points2ClustersDict)

plotDenLACResult(points2ClustersDict)