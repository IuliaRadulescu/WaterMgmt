import numpy as np
import pandas as pd
import os
import math
from math import radians, degrees
import itertools
from sklearn.metrics.pairwise import haversine_distances

import utils
import plotTrajectoriesHYSPLIT
import denlac

def bearing(lat1, lon1, lat2, lon2):

    return np.arctan2((np.cos((lat2)) * np.sin((lon2-lon1))),
        (np.cos((lat1)) * np.sin((lat2)) - np.sin((lat1)) * np.cos((lat2)) * np.cos((lon2-lon1))))

    

def generateDenLACCoords(distance_type):

    trajDf = utils.readTraj()

    timeDf = trajDf[['year', 'month', 'day', 'hour', 'minute']]
    timeDf = timeDf.astype(str)
    dateTimeDf = pd.to_datetime(timeDf['year'] + timeDf['month'] + timeDf['day'] + timeDf['hour'] + timeDf['minute'], format='%y%m%d%H%M')
    trajDf['dateTime'] = dateTimeDf

    trajDf['lat_r'] = trajDf.lat.apply(radians)
    trajDf['lon_r'] = trajDf.lon.apply(radians)

    groupedByTraj = trajDf.groupby('ntra')

    trajectoryDict = {}

    for ntra, group in groupedByTraj:
        g = group.sort_values('dateTime', ascending=False)

        if (distance_type == 'bearing'):
            g['bearing'] = [bearing(lat1, lon1, lat2, lon2) if math.isnan(bearing(lat1, lon1, lat2, lon2)) == False else 0 \
                                for (lat1, lon1), (lat2, lon2) in \
                                zip(g[['lat_r', 'lon_r']].values, g[['lat_r', 'lon_r']].shift(-1).values)]
            trajectoryDict[ntra-1] = np.array(g['bearing'])
        elif (distance_type == 'haversine'):
            g['haversine'] = haversine_distances(g[['lat_r', 'lon_r']].values, np.zeros((1,2))) * 6371000/1000
            trajectoryDict[ntra-1] = np.array(g['haversine'])
        elif (distance_type == 'euclidean'):
            trajectoryDict[ntra-1] = np.array(g[['lat_r', 'lon_r']])
        else:
            g['haversine'] = haversine_distances(g[['lat_r', 'lon_r']].values, np.zeros((1,2))) * 6371000/1000
            trajectoryDict[ntra-1] = np.array(g['haversine'])

    trajectoryLens = []

    for key, elem in trajectoryDict.items():
        trajectoryLens.append(np.shape(elem)[0])

    minLen = min(trajectoryLens)

    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    fileLocation = scriptDirectory + '/trajectories/czech_june_2021/'
    trajDenLACFile = open(fileLocation + 'trajsDenLAC_' + distance_type + '.txt', 'w')

    dataset = []

    for key, elem in trajectoryDict.items():
        firstMinLen = elem[0:minLen]

        listToAppend = firstMinLen.tolist()
        listToAppend.extend([key])
        dataset.append(listToAppend)

        line = ','.join(map(str, firstMinLen))
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

    return [(point2pointId[point], point2clusterId[point]) for point in point2pointId.keys()]

def plotDenLACResult(denLACResult):

    trajDf = utils.readTraj()
    groupedByTraj = trajDf.groupby('ntra')

    for ntra, group in groupedByTraj:
        group['labelDenLAC'] = [denLACResult[ntra-1]] * group['label'].shape[0]
        if ntra == 1:
            resultDf = group
        else:
            resultDf = pd.concat([resultDf, group], ignore_index=True, sort=False)

    plotTrajectoriesHYSPLIT.plotTraj(resultDf, 'labelDenLAC')

datasetWithLabels = generateDenLACCoords('euclidean')

dataset = np.array([elem[0:-1] for elem in datasetWithLabels])

print('Dataset of shape', np.shape(dataset))

joinedPartitions = denlac.runDenLAC(dataset)

points2ClustersDict = dict(getClustersForDatasetElements(datasetWithLabels, joinedPartitions))

print(points2ClustersDict)

plotDenLACResult(points2ClustersDict)