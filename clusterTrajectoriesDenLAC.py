import collections
import os
import random

import numpy as np
import pandas as pd
from math import radians, degrees, floor, atan
import matplotlib.pyplot as plt
import utm

import utils
import plotTrajectoriesHYSPLIT
import denlac

def convertToCartesian(elem):

    x, y, _, _ = utm.from_latlon(elem[0], elem[1])
    
    return (x, y)

def normalize(x, y, start, end):
    width = end - start
    x = (x - x.min())/(x.max() - x.min()) * width + start
    y = (y - y.min())/(y.max() - y.min()) * width + start

    return (x, y)

def translateToOrigin(x, y, x0, y0):
    x = (x - x0)
    y = (y - y0)
    return (x, y)

def extractAngles(trajList):

    trajectoryAngles = []

    for trajId in range(len(trajList)):
        angleList = [computeRelativeAngle(x, y) for (x, y) in trajList[trajId]]
        trajectoryAngles.append(elementsListRepresentatives(angleList))

    return trajectoryAngles

def computeRelativeAngle(x, y):
    
    angle = round(degrees(atan(abs(y/x))))

    relativeAngle = angle

    if (x > 0 and y < 0):
        relativeAngle = 90 + angle

    if (x > 0 and y == 0):
        relativeAngle = 0

    if (x > 0 and y > 0):
        relativeAngle = 270 + angle

    if (x == 0 and y < 0):
        relativeAngle = 180

    if (x == 0 and y > 0):
        relativeAngle = 90

    if (x < 0 and y < 0):
        relativeAngle = 270 - angle

    if (x < 0 and y == 0):
        relativeAngle = 270

    return relativeAngle

'''
elementsList: 1-d numpy array
'''
def elementsListRepresentatives(elementsList):

    mean = round(sum(elementsList)/len(elementsList), 2)
    median = np.median(elementsList)

    return [mean, median]

def computeTrajDict(groupedByTraj):

    trajectories = []
    trajectoryLens = []

    for _, group in groupedByTraj:
        latLonArray = np.array(group[['lat_r', 'lon_r']])
        cartesianArray = np.array([convertToCartesian(elem) for elem in latLonArray])
        trajectories.append(cartesianArray)
        trajectoryLens.append(len(cartesianArray))

    minLen = min(trajectoryLens)

    for trajectoryId in range(len(trajectories)):
        trajectories[trajectoryId] = trajectories[trajectoryId][0:minLen]

    trajectories = np.array(trajectories)

    # normalize trajectories

    xValues = trajectories[:,:,0].flatten()
    yValues = trajectories[:,:,1].flatten()

    (xValues, yValues) = normalize(xValues, yValues, 1, 10)

    (xValues, yValues) = translateToOrigin(xValues, yValues, xValues[0], yValues[0])
 
    trajectoryDict = collections.defaultdict(list)
    ntra = 0

    for xyId in range(len(xValues)):
        ntra = floor(xyId / minLen)
        trajectoryDict[ntra].append((round(xValues[xyId], 5), round(yValues[xyId], 5)))

    return trajectoryDict


def generateDenLACCoords(distance_type):

    trajDf = utils.readTraj()

    trajDf['lat_r'] = trajDf.lat.apply(radians)
    trajDf['lon_r'] = trajDf.lon.apply(radians)

    groupedByTraj = trajDf.groupby('ntra')

    trajectoryDict = computeTrajDict(groupedByTraj)

    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    fileLocation = scriptDirectory + '/trajectories/czech_june_2021/'
    trajDenLACFile = open(fileLocation + 'trajsDenLAC_' + distance_type + '.txt', 'w')

    dataset = []

    for key, elem in trajectoryDict.items():
        angleList = [computeRelativeAngle(x, y) for (x, y) in elem[1:]]
        listToAppend = elementsListRepresentatives(angleList)
        listToAppend.extend([int(key)])
        dataset.append(listToAppend)

        line = ','.join(map(str, elem))
        line += ',' + str(key) + '\n'
        trajDenLACFile.write(line)

    trajDenLACFile.close()

    return dataset


def getClustersForDatasetElements(datasetWithLabels, clusterPoints):
        
    trajectoryRepresentative2PointId = {}
    trajectoryRepresentative2ClusterId = {}

    for trajectoryRepresentatives in datasetWithLabels:
        trajectoryRepresentatives = np.array(trajectoryRepresentatives)
        index = trajectoryRepresentatives[0:-1]
        trajectoryRepresentative2PointId[tuple(index)] = int(trajectoryRepresentatives[-1])

    for clusterId, elementsInCluster in clusterPoints.items():
        elementsInCluster = np.array(elementsInCluster)
        for element in elementsInCluster:
            trajectoryRepresentative2ClusterId[tuple(element)] = clusterId

    return [(trajectoryRepresentative, trajectoryRepresentative2ClusterId[trajectoryRepresentative]) if trajectoryRepresentative in trajectoryRepresentative2ClusterId.keys() else (trajectoryRepresentative2PointId[trajectoryRepresentative], -1) for trajectoryRepresentative in trajectoryRepresentative2PointId.keys()]

def getClustersForTrajectories(trajectoryRepresentative2ClusterId):

    print(trajectoryRepresentative2ClusterId)

    trajectoryId2ClusterId = {}

    trajDf = utils.readTraj()

    trajDf['lat_r'] = trajDf.lat.apply(radians)
    trajDf['lon_r'] = trajDf.lon.apply(radians)

    groupedByTraj = trajDf.groupby('ntra')

    trajectoryDict = computeTrajDict(groupedByTraj)

    for key, elem in trajectoryDict.items():
        trajectoryRepresentatives = tuple(elementsListRepresentatives([computeRelativeAngle(x, y) for (x, y) in elem[1:]]))
        if (trajectoryRepresentatives in trajectoryRepresentative2ClusterId):
            trajectoryId2ClusterId[key] = trajectoryRepresentative2ClusterId[trajectoryRepresentatives]

    return trajectoryId2ClusterId

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

    clusters2representatives = collections.defaultdict(list)

    for representatives, cluster in denLACResult.items():
        clusters2representatives[cluster].append(representatives)

    trajDf = utils.readTraj()

    trajDf['lat_r'] = trajDf.lat.apply(radians)
    trajDf['lon_r'] = trajDf.lon.apply(radians)

    groupedByTraj = trajDf.groupby('ntra')

    trajectoryDict = computeTrajDict(groupedByTraj)

    representatives2Trajectories = {}

    for _, elem in trajectoryDict.items():
        angleList = [computeRelativeAngle(x, y) for (x, y) in elem[1:]]
        representatives = tuple(elementsListRepresentatives(angleList))
        representatives2Trajectories[representatives] = elem
        
    nrColors = len(set(denLACResult.values()))

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(nrColors)]

    for representatives in representatives2Trajectories:
        if representatives not in denLACResult:
            continue
        traj = np.array(representatives2Trajectories[representatives])
        plt.plot(traj[:, 0], traj[:, 1], label = 'traj ' + str(representatives), color = colors[denLACResult[representatives]])

    plt.show()


datasetWithLabels = generateDenLACCoords('euclidean')

dataset = [elem[0:-1] for elem in datasetWithLabels]

joinedPartitions = denlac.runDenLAC(dataset)

points2ClustersDict = dict(getClustersForDatasetElements(datasetWithLabels, joinedPartitions))

plotPlaneProjection(points2ClustersDict)

trajectoryId2ClusterId = getClustersForTrajectories(points2ClustersDict)

plotDenLACResult(trajectoryId2ClusterId)