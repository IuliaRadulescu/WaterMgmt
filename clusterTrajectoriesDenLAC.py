import collections
import os
import random

import numpy as np
import pandas as pd
from math import radians, degrees, floor, atan
import matplotlib.pyplot as plt
from similaritymeasures import frechet_dist
from sklearn.metrics import silhouette_score
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
        
        angle = round(degrees(atan(abs(y/x))), 3)

        relativeAngle = angle

        if (x > 0 and y > 0):
            relativeAngle = 90 - angle

        if (x > 0 and y < 0):
            relativeAngle = 90 + angle

        if (x > 0 and y == 0):
            relativeAngle = 90

        if (x == 0 and y > 0):
            relativeAngle = 0

        if (x == 0 and y < 0):
            relativeAngle = 180

        if (x < 0 and y < 0):
            relativeAngle = 270 - angle

        if (x < 0 and y > 0):
            relativeAngle = 270 + angle

        if (x < 0 and y == 0):
            relativeAngle = 270

        return relativeAngle

    '''
    elementsList: 1-d numpy array
    '''
    @staticmethod
    def elementsListRepresentatives(elementsList):

        minVal = min(elementsList)
        maxVal = max(elementsList)

        p1 = 25
        p2 = 50
        p3 = 75

        q1 = np.percentile(elementsList,  p1)
        q2 = np.percentile(elementsList,  p2)
        q3 = np.percentile(elementsList,  p3)

        return [minVal, q1, q2, q3, maxVal]

class TrajectoryClusterer:

    def __init__(self, trajectories):
        
        self.trajectories = trajectories
        self.adaptedTrajectoriesDict = self.computeAdaptedTrajDict()
        self.trajectoriesDict = self.computeTrajDict()

    def getAdaptedTrajDict(self):
        return self.adaptedTrajectoriesDict

    def getTrajDict(self):
        return self.trajectoriesDict

    def computeTrajDict(self):

        trajectoryDict = {}
        trajectoryLens = []

        for ntra, group in self.trajectories:
            trajectoryDict[ntra-1] = np.array(group[['lat_r', 'lon_r']])
            trajectoryLens.append(len(trajectoryDict[ntra-1]))

        minLen = min(trajectoryLens)

        for trajectoryId in trajectoryDict:
            trajectoryDict[trajectoryId] = trajectoryDict[trajectoryId][0:minLen]

        return trajectoryDict

    '''
    computes trajctories converted to catesian, normalized and translated to origin
    '''
    def computeAdaptedTrajDict(self):

        cartesianTrajectories = []
        trajectoryLens = []

        for _, group in self.trajectories:
            latLonArray = np.array(group[['lat_r', 'lon_r']])
            cartesianArray = np.array([TrajectoryUtils.convertToCartesian(elem) for elem in latLonArray])
            cartesianTrajectories.append(cartesianArray)
            trajectoryLens.append(len(cartesianArray))

        minLen = min(trajectoryLens)

        for trajectoryId in range(len(cartesianTrajectories)):
            cartesianTrajectories[trajectoryId] = cartesianTrajectories[trajectoryId][0:minLen]

        cartesianTrajectories = np.array(cartesianTrajectories)

        # normalize trajectories

        xValues = cartesianTrajectories[:,:,0].flatten()
        yValues = cartesianTrajectories[:,:,1].flatten()

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

        for key, elem in self.adaptedTrajectoriesDict.items():
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
        trajectory2ClusterId = {}
        clusterId2Trajectory = collections.defaultdict(list)

        for key, elem in self.adaptedTrajectoriesDict.items():
            trajectoryRepresentatives = tuple(TrajectoryUtils.elementsListRepresentatives([TrajectoryUtils.computeRelativeAngle(x, y) for (x, y) in elem[1:]]))
            if (trajectoryRepresentatives in trajectoryRepresentative2ClusterId):
                clusterId = trajectoryRepresentative2ClusterId[trajectoryRepresentatives]
                trajectoryId2ClusterId[key] = clusterId
                trajectory2ClusterId[tuple(elem)] = clusterId
                clusterId2Trajectory[clusterId].append(elem)

        return (clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId)

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

class TrajectoryEvaluation:

    def distanceFrechet(self, traj1, traj2):
        return frechet_dist(traj1, traj2)

    def getTrajectoryClusterCentroid(self, trajs):

        noDims = np.shape(trajs)[1]
        trajsNp = np.array(trajs)

        centroid = []
        
        for d in range(noDims):
            centroid.append(np.mean(trajsNp[:,d,:], axis=0))

        return centroid

    def computeDaviesBouldin(self, clusterId2Trajectory):

        def getClusterAvg(trajs, centroid):
            distances = []

            for traj in trajs:
                distances.append(self.distanceFrechet(traj, centroid))

            return sum(distances)/len(distances)

        clusterIds2Centroids = {}
        clusterIds2Avgs = {}

        maximumsSum = 0

        for clusterId1 in clusterId2Trajectory:
            maxValue = 0
            for clusterId2 in clusterId2Trajectory:
                if clusterId1 > clusterId2:
                    continue

                if clusterId1 not in clusterIds2Centroids:
                    clusterIds2Centroids[clusterId1] = self.getTrajectoryClusterCentroid(clusterId2Trajectory[clusterId1])
                if clusterId2 not in clusterIds2Centroids:
                    clusterIds2Centroids[clusterId2] = self.getTrajectoryClusterCentroid(clusterId2Trajectory[clusterId2])

                centroid1 = clusterIds2Centroids[clusterId1]
                centroid2 = clusterIds2Centroids[clusterId2]

                if clusterId1 not in clusterIds2Avgs:
                    clusterIds2Avgs[clusterId1] = getClusterAvg(clusterId2Trajectory[clusterId1], centroid1)
                if clusterId2 not in clusterIds2Avgs:
                    clusterIds2Avgs[clusterId2] = getClusterAvg(clusterId2Trajectory[clusterId2], centroid2)

                cluster1Avg = clusterIds2Avgs[clusterId1]
                cluster2Avg = clusterIds2Avgs[clusterId2]

                distHauss = self.distanceFrechet(centroid1, centroid2)

                dbValue = (cluster1Avg + cluster2Avg) / (distHauss) if distHauss > 0 else 0

                if (dbValue > maxValue):
                    maxValue = dbValue

            maximumsSum += maxValue

        return maximumsSum/len(set(clusterId2Trajectory.keys()))

    def computeCalinskiHarabasz(self, clusterId2Trajectory):

        clustersNr = len(set(clusterId2Trajectory.keys()))
        allTrajectories = [traj for trajList in clusterId2Trajectory.values() for traj in trajList]
        trajectoriesNr = len(allTrajectories)

        datasetCentroid = self.getTrajectoryClusterCentroid(allTrajectories)

        sum1 = 0
        clusterIds2Centroids = {}

        for clusterId in clusterId2Trajectory:
            elementsInCluster = len(clusterId2Trajectory[clusterId])

            if clusterId not in clusterIds2Centroids:
                clusterIds2Centroids[clusterId] = self.getTrajectoryClusterCentroid(clusterId2Trajectory[clusterId])

            distHaussCentroid = self.distanceFrechet(clusterIds2Centroids[clusterId], datasetCentroid)

            sum1 += elementsInCluster * distHaussCentroid

        term1 = sum1/(clustersNr - 1)

        sum2 = 0
        for clusterId in clusterId2Trajectory:
            for traj in clusterId2Trajectory[clusterId]:
                sum2 += self.distanceFrechet(traj, clusterIds2Centroids[clusterId])

        term2 = sum2/(trajectoriesNr - clustersNr)

        return term1/term2

    def computeSilhouette(self, trajectory2ClusterId):

        labels = list(trajectory2ClusterId.values())
        allTrajectories = list(trajectory2ClusterId.keys())
        
        distanceMatrix = []

        for traj1 in allTrajectories:
            matrixRow = []
            for traj2 in allTrajectories:
                matrixRow.append(self.distanceFrechet(traj1, traj2))
            distanceMatrix.append(matrixRow)

        return silhouette_score(X=distanceMatrix, labels=labels, metric='precomputed')


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
resultsPlotter.plotPlaneProjection(points2ClustersDict, trajectoryClusterer.getAdaptedTrajDict())
clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId = trajectoryClusterer.getClustersForTrajectories(points2ClustersDict)
resultsPlotter.plotDenLACResult(trajectoryId2ClusterId)

trajectoryEvaluation = TrajectoryEvaluation()
dbI = trajectoryEvaluation.computeDaviesBouldin(clusterId2Trajectory)
chI = trajectoryEvaluation.computeCalinskiHarabasz(clusterId2Trajectory)
silh = trajectoryEvaluation.computeSilhouette(trajectory2ClusterId)

print('Davies Bouldin Index', dbI)
print('Calinski Harabasz Index', chI)
print('Silhouette Index', silh)