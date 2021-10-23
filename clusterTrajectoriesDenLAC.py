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

class TrajectoryUtils:

    @staticmethod
    def convertToCartesian(elem):

        x, y, _, _ = utm.from_latlon(elem[0], elem[1])
        
        return (x, y)

    @staticmethod
    def normalize(x, y, start, end):
        width = end - start
        x = (x - x.min())/(x.max() - x.min()) * width + start
        y = (y - y.min())/(y.max() - y.min()) * width + start

        return (x, y)

    @staticmethod
    def translateToOrigin(x, y, x0, y0):
        x = (x - x0)
        y = (y - y0)
        return (x, y)

    @staticmethod
    def extractAngles(trajList):

        trajectoryAngles = []

        for trajId in range(len(trajList)):
            angleList = [TrajectoryUtils.computeRelativeAngle(x, y) for (x, y) in trajList[trajId]]
            trajectoryAngles.append(TrajectoryUtils.elementsListRepresentatives(angleList))

        return trajectoryAngles

    @staticmethod
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
    @staticmethod
    def elementsListRepresentatives(elementsList):

        mean = round(sum(elementsList)/len(elementsList), 2)
        median = np.median(elementsList)

        return [mean, median]

class TrajectoryClusterer:

    def __init__(self, trajectories):
        
        self.trajectories = trajectories
        self.trajectoryDict = self.computeTrajDict()

    def getTrajectoryDict(self):
        return self.trajectoryDict

    def computeTrajDict(self):

        trajectories = []
        trajectoryLens = []

        for _, group in self.trajectories:
            latLonArray = np.array(group[['lat_r', 'lon_r']])
            cartesianArray = np.array([TrajectoryUtils.convertToCartesian(elem) for elem in latLonArray])
            trajectories.append(cartesianArray)
            trajectoryLens.append(len(cartesianArray))

        minLen = min(trajectoryLens)

        for trajectoryId in range(len(trajectories)):
            trajectories[trajectoryId] = trajectories[trajectoryId][0:minLen]

        trajectories = np.array(trajectories)

        # normalize trajectories

        xValues = trajectories[:,:,0].flatten()
        yValues = trajectories[:,:,1].flatten()

        (xValues, yValues) = TrajectoryUtils.normalize(xValues, yValues, 1, 10)

        (xValues, yValues) = TrajectoryUtils.translateToOrigin(xValues, yValues, xValues[0], yValues[0])
    
        trajectoryDict = collections.defaultdict(list)
        ntra = 0

        for xyId in range(len(xValues)):
            ntra = floor(xyId / minLen)
            trajectoryDict[ntra].append((round(xValues[xyId], 5), round(yValues[xyId], 5)))

        return trajectoryDict


    def generateDenLACCoords(self, filename, foldername):

        scriptDirectory = os.path.dirname(os.path.abspath(__file__))
        fileLocation = scriptDirectory + '/trajectories/' + foldername + '/'
        trajDenLACFile = open(fileLocation + filename, 'w')

        dataset = []

        for key, elem in self.trajectoryDict.items():
            angleList = [TrajectoryUtils.computeRelativeAngle(x, y) for (x, y) in elem[1:]]
            listToAppend = TrajectoryUtils.elementsListRepresentatives(angleList)
            listToAppend.extend([int(key)])
            dataset.append(listToAppend)

            line = ','.join(map(str, elem))
            line += ',' + str(key) + '\n'
            trajDenLACFile.write(line)

        trajDenLACFile.close()

        return dataset


    def getClustersForDatasetElements(self, datasetWithLabels, clusterPoints):
            
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

    def getClustersForTrajectories(self, trajectoryRepresentative2ClusterId):

        trajectoryId2ClusterId = {}

        for key, elem in self.trajectoryDict.items():
            trajectoryRepresentatives = tuple(TrajectoryUtils.elementsListRepresentatives([TrajectoryUtils.computeRelativeAngle(x, y) for (x, y) in elem[1:]]))
            if (trajectoryRepresentatives in trajectoryRepresentative2ClusterId):
                trajectoryId2ClusterId[key] = trajectoryRepresentative2ClusterId[trajectoryRepresentatives]

        return trajectoryId2ClusterId

class ResultsPlotter:

    def __init__(self, trajectories):
        self.trajectories = trajectories

    def plotDenLACResult(self, clusteringResult):

        for ntra, group in self.trajectories:
            if (ntra-1) not in clusteringResult:
                continue
            group['labelDenLAC'] = [clusteringResult[ntra-1]] * group['label'].shape[0]
            if ntra == 1:
                resultDf = group
            else:
                resultDf = pd.concat([resultDf, group], ignore_index=True, sort=False)

        plotTrajectoriesHYSPLIT.plotTraj(resultDf, 'labelDenLAC')

    def plotPlaneProjection(self, denLACResult, trajectoryDict):

        clusters2representatives = collections.defaultdict(list)

        for representatives, cluster in denLACResult.items():
            clusters2representatives[cluster].append(representatives)

        representatives2Trajectories = {}

        for _, elem in trajectoryDict.items():
            angleList = [TrajectoryUtils.computeRelativeAngle(x, y) for (x, y) in elem[1:]]
            representatives = tuple(TrajectoryUtils.elementsListRepresentatives(angleList))
            representatives2Trajectories[representatives] = elem
            
        nrColors = len(set(denLACResult.values()))

        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(nrColors)]

        for representatives in representatives2Trajectories:
            if representatives not in denLACResult:
                continue
            traj = np.array(representatives2Trajectories[representatives])
            plt.plot(traj[:, 0], traj[:, 1], label = 'traj ' + str(representatives), color = colors[denLACResult[representatives]])

        plt.show()


trajDf = utils.readTraj()

trajDf['lat_r'] = trajDf.lat.apply(radians)
trajDf['lon_r'] = trajDf.lon.apply(radians)

trajectories = trajDf.groupby('ntra')

trajectoryClusterer = TrajectoryClusterer(trajectories)
datasetWithLabels = trajectoryClusterer.generateDenLACCoords('trajsDenLAC.txt', 'czech_june_2021')
dataset = [elem[0:-1] for elem in datasetWithLabels]
joinedPartitions = denlac.runDenLAC(dataset)

points2ClustersDict = dict(trajectoryClusterer.getClustersForDatasetElements(datasetWithLabels, joinedPartitions))
resultsPlotter = ResultsPlotter(trajectories)
resultsPlotter.plotPlaneProjection(points2ClustersDict, trajectoryClusterer.getTrajectoryDict())
trajectoryId2ClusterId = trajectoryClusterer.getClustersForTrajectories(points2ClustersDict)
resultsPlotter.plotDenLACResult(trajectoryId2ClusterId)