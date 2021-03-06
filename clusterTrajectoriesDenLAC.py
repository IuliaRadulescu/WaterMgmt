import collections
import os
import random
from random import randrange

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
            trajVector = TrajectoryUtils.computeTrajVector(trajList[trajId])
            angleList = [TrajectoryUtils.computeRelativeAngle(x, y) for (x, y) in trajVector]
            trajectoryAngles.append(TrajectoryUtils.elementsListRepresentatives(angleList))

        return trajectoryAngles

    @staticmethod
    def computeRelativeAngle(x, y):
        
        angle = round(degrees(atan(abs(y/x))))

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

        p1 = 25
        p2 = 50
        p3 = 75

        q1 = np.percentile(elementsList,  p1)
        q2 = np.percentile(elementsList,  p2)
        q3 = np.percentile(elementsList,  p3)

        return [q1, q2, q3]

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

    def getClustersForTrajectoriesUsingRepresentatives(self, trajectoryRepresentative2ClusterId):

        trajectoryId2ClusterId = {}
        trajectory2ClusterId = {}
        clusterId2Trajectory = collections.defaultdict(list)

        for key, elem in self.adaptedTrajectoriesDict.items():
            trajectoryRepresentatives = tuple(TrajectoryUtils.elementsListRepresentatives([TrajectoryUtils.computeRelativeAngle(x, y) for (x, y) in elem[1:]]))
            if (trajectoryRepresentatives in trajectoryRepresentative2ClusterId):
                clusterId = trajectoryRepresentative2ClusterId[trajectoryRepresentatives]
                trajectoryId2ClusterId[key] = clusterId
                trajectory2ClusterId[tuple(trajectoryRepresentatives)] = clusterId
                clusterId2Trajectory[clusterId].append(trajectoryRepresentatives)

        return (clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId)

    def getWorstCaseClusters(self, clustersNr = 2):

        trajectoryId2ClusterId = {}
        trajectory2ClusterId = {}
        clusterId2Trajectory = collections.defaultdict(list)

        clusterId = 0

        for key, elem in self.adaptedTrajectoriesDict.items():
            clusterId = clusterId + 1 if clusterId < clustersNr else clustersNr
            trajectoryId2ClusterId[key] = clusterId
            trajectory2ClusterId[tuple(elem)] = clusterId
            clusterId2Trajectory[clusterId].append(elem)
                
        return (clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId)

    def getWorstCaseClustersUsingRepresentatives(self, trajectoryRepresentative2ClusterId, clustersNr = 2):

        trajectoryId2ClusterId = {}
        trajectory2ClusterId = {}
        clusterId2Trajectory = collections.defaultdict(list)

        clusterId = 0

        for key, elem in self.adaptedTrajectoriesDict.items():
            trajectoryRepresentatives = tuple(TrajectoryUtils.elementsListRepresentatives([TrajectoryUtils.computeRelativeAngle(x, y) for (x, y) in elem[1:]]))
            clusterId = clusterId + 1 if clusterId < clustersNr else clustersNr
            trajectoryId2ClusterId[key] = clusterId
            trajectory2ClusterId[tuple(trajectoryRepresentatives)] = clusterId
            clusterId2Trajectory[clusterId].append(trajectoryRepresentatives)
                
        return (clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId)

    def getRandomClustersUsingRepresentatives(self, trajectoryRepresentative2ClusterId, clustersNr = 2):

        trajectoryId2ClusterId = {}
        trajectory2ClusterId = {}
        clusterId2Trajectory = collections.defaultdict(list)

        for key, elem in self.adaptedTrajectoriesDict.items():
            trajectoryRepresentatives = tuple(TrajectoryUtils.elementsListRepresentatives([TrajectoryUtils.computeRelativeAngle(x, y) for (x, y) in elem[1:]]))
            clusterId = randrange(clustersNr)
            trajectoryId2ClusterId[key] = clusterId
            trajectory2ClusterId[tuple(trajectoryRepresentatives)] = clusterId
            clusterId2Trajectory[clusterId].append(trajectoryRepresentatives)
                
        return (clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId)

    def getRandomClusters(self, clustersNr = 2):

        trajectoryId2ClusterId = {}
        trajectory2ClusterId = {}
        clusterId2Trajectory = collections.defaultdict(list)

        for key, elem in self.adaptedTrajectoriesDict.items():
            clusterId = randrange(clustersNr)
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

    @staticmethod
    def angleToQuadrant(degree):
        if degree <= 90:
            return 1
        else:
            if degree <= 180:
                return 2
            else:
                if degree <= 270:
                    return 3
                else:
                    if degree <= 360:
                        return 4
    @staticmethod
    def reduceToQuadrant(angle, quad):

        if (quad == 2):
            angle -= 90
            angle *= -1
        
        if (quad == 3):
            angle = 270 - angle

        if (quad == 4):
            angleX = 360 - angle
            angleX *= -1

        return angle

    @staticmethod
    def DistFunc(x, y):

        noDims = len(x)

        angleDiffs = []
        for dim in range(noDims):

            angleX = x[dim]
            angleY = y[dim]

            quadX = TrajectoryEvaluation.angleToQuadrant(angleX)
            quadY = TrajectoryEvaluation.angleToQuadrant(angleY)

            # reduce to quadrant
            angleX = TrajectoryEvaluation.reduceToQuadrant(angleX, quadX)
            angleY = TrajectoryEvaluation.reduceToQuadrant(angleY, quadX)

            # if angles are in the same quadrant just take the difference between abs
            if (quadX == quadY):
                angleDiff = abs(abs(angleX) - abs(angleY))
            # if adjacent quadrants
            elif ((quadX != quadY) and (quadX % 2) == (quadY % 2)):
                angleDiff = abs(abs(angleX) + abs(angleY))
            # if opposed quadrants
            elif ((quadX != quadY) and (quadX % 2) != (quadY % 2)):
                angleDiff = 360 - abs(abs(angleX) + abs(angleY))

            angleDiffs.append(abs(angleDiff))

        angleDiffsSum = sum(angleDiffs)

        return angleDiffsSum/noDims


    def trajectoriesDistance(self, traj1, traj2, useRepresentativesDist = False):
        return frechet_dist(traj1, traj2) if useRepresentativesDist == False else TrajectoryEvaluation.DistFunc(traj1, traj2)

    def getTrajectoryClusterCentroid(self, trajs, useRepresentativesDist = False):

        trajsNp = np.array(trajs)
        noDims = np.shape(trajs)[1]

        centroid = []
        
        for d in range(noDims):
            correctProjection = trajsNp[:,d,:] if useRepresentativesDist == False else trajsNp[:,d]
            centroid.append(np.mean(correctProjection, axis=0))

        return centroid

    def computeDaviesBouldin(self, clusterId2Trajectory, useRepresentativesDist = False):

        def getClusterAvg(trajs, centroid, useRepresentativesDist):
            distances = []

            for traj in trajs:
                distances.append(self.trajectoriesDistance(traj, centroid, useRepresentativesDist))

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
                    clusterIds2Centroids[clusterId1] = self.getTrajectoryClusterCentroid(clusterId2Trajectory[clusterId1], useRepresentativesDist)
                if clusterId2 not in clusterIds2Centroids:
                    clusterIds2Centroids[clusterId2] = self.getTrajectoryClusterCentroid(clusterId2Trajectory[clusterId2], useRepresentativesDist)

                centroid1 = clusterIds2Centroids[clusterId1]
                centroid2 = clusterIds2Centroids[clusterId2]

                if clusterId1 not in clusterIds2Avgs:
                    clusterIds2Avgs[clusterId1] = getClusterAvg(clusterId2Trajectory[clusterId1], centroid1, useRepresentativesDist)
                if clusterId2 not in clusterIds2Avgs:
                    clusterIds2Avgs[clusterId2] = getClusterAvg(clusterId2Trajectory[clusterId2], centroid2, useRepresentativesDist)

                cluster1Avg = clusterIds2Avgs[clusterId1]
                cluster2Avg = clusterIds2Avgs[clusterId2]

                distFrech = self.trajectoriesDistance(centroid1, centroid2, useRepresentativesDist)

                dbValue = (cluster1Avg + cluster2Avg) / (distFrech) if distFrech > 0 else 0

                if (dbValue > maxValue):
                    maxValue = dbValue

            maximumsSum += maxValue

        return maximumsSum/len(set(clusterId2Trajectory.keys()))

    def computeCalinskiHarabasz(self, clusterId2Trajectory, useRepresentativesDist = False):

        clustersNr = len(set(clusterId2Trajectory.keys()))
        allTrajectories = [traj for trajList in clusterId2Trajectory.values() for traj in trajList]
        trajectoriesNr = len(allTrajectories)

        datasetCentroid = self.getTrajectoryClusterCentroid(allTrajectories, useRepresentativesDist)

        sum1 = 0
        clusterIds2Centroids = {}

        for clusterId in clusterId2Trajectory:
            elementsInCluster = len(clusterId2Trajectory[clusterId])

            if clusterId not in clusterIds2Centroids:
                clusterIds2Centroids[clusterId] = self.getTrajectoryClusterCentroid(clusterId2Trajectory[clusterId], useRepresentativesDist)

            distHaussCentroid = self.trajectoriesDistance(clusterIds2Centroids[clusterId], datasetCentroid, useRepresentativesDist)

            sum1 += elementsInCluster * distHaussCentroid

        term1 = sum1/(clustersNr - 1)

        sum2 = 0
        for clusterId in clusterId2Trajectory:
            for traj in clusterId2Trajectory[clusterId]:
                sum2 += self.trajectoriesDistance(traj, clusterIds2Centroids[clusterId], useRepresentativesDist)

        term2 = sum2/(trajectoriesNr - clustersNr)

        return term1/term2

    def computeSilhouette(self, trajectory2ClusterId, useRepresentativesDist = False):

        labels = list(trajectory2ClusterId.values())
        allTrajectories = list(trajectory2ClusterId.keys())
        
        distanceMatrix = []

        for traj1 in allTrajectories:
            matrixRow = []
            for traj2 in allTrajectories:
                matrixRow.append(self.trajectoriesDistance(traj1, traj2, useRepresentativesDist))
            distanceMatrix.append(matrixRow)

        return silhouette_score(X=distanceMatrix, labels=labels, metric='precomputed')

    @staticmethod
    def printEvaluationMeasures(clusterId2Trajectory, trajectory2ClusterId, useRepresentativesDist = False):
        
        print('')
        print('EVALUATION MEASURES', len(set(clusterId2Trajectory.keys())), 'clusters')
        print('')

        dbI = trajectoryEvaluation.computeDaviesBouldin(clusterId2Trajectory, useRepresentativesDist)
        chI = trajectoryEvaluation.computeCalinskiHarabasz(clusterId2Trajectory, useRepresentativesDist)
        silh = trajectoryEvaluation.computeSilhouette(trajectory2ClusterId, useRepresentativesDist)

        print('Davies Bouldin Index', dbI)
        print('Calinski Harabasz Index', chI)
        print('Silhouette Index', silh)


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

print('=== === WITH REPRESENTATIVES === ===')

clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId = trajectoryClusterer.getClustersForTrajectoriesUsingRepresentatives(points2ClustersDict)

print('=== RESULTS ===')

trajectoryEvaluation = TrajectoryEvaluation()
TrajectoryEvaluation.printEvaluationMeasures(clusterId2Trajectory, trajectory2ClusterId, True)

print('=== EDGE CASES ===')

print('2 clusters')
# worst case indices (each trajectory in its separate cluster)
clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId = trajectoryClusterer.getWorstCaseClustersUsingRepresentatives(points2ClustersDict, 2)
TrajectoryEvaluation.printEvaluationMeasures(clusterId2Trajectory, trajectory2ClusterId, True)

clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId = trajectoryClusterer.getRandomClustersUsingRepresentatives(points2ClustersDict, 2)
TrajectoryEvaluation.printEvaluationMeasures(clusterId2Trajectory, trajectory2ClusterId, True)

print('3 clusters')
clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId = trajectoryClusterer.getWorstCaseClustersUsingRepresentatives(points2ClustersDict, 3)
TrajectoryEvaluation.printEvaluationMeasures(clusterId2Trajectory, trajectory2ClusterId, True)

clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId = trajectoryClusterer.getRandomClustersUsingRepresentatives(points2ClustersDict, 3)
TrajectoryEvaluation.printEvaluationMeasures(clusterId2Trajectory, trajectory2ClusterId, True)

print('=== === CLASSIC === ===')

print('=== RESULTS ===')

clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId = trajectoryClusterer.getClustersForTrajectories(points2ClustersDict)
trajectoryEvaluation = TrajectoryEvaluation()
TrajectoryEvaluation.printEvaluationMeasures(clusterId2Trajectory, trajectory2ClusterId)

print('=== EDGE CASES ===')

print('2 clusters')
# worst case indices (each trajectory in its separate cluster)
clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId = trajectoryClusterer.getWorstCaseClusters(2)
TrajectoryEvaluation.printEvaluationMeasures(clusterId2Trajectory, trajectory2ClusterId)

clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId = trajectoryClusterer.getRandomClusters(2)
TrajectoryEvaluation.printEvaluationMeasures(clusterId2Trajectory, trajectory2ClusterId)

print('3 clusters')
clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId = trajectoryClusterer.getWorstCaseClusters(3)
TrajectoryEvaluation.printEvaluationMeasures(clusterId2Trajectory, trajectory2ClusterId)

clusterId2Trajectory, trajectory2ClusterId, trajectoryId2ClusterId = trajectoryClusterer.getRandomClusters(3)
TrajectoryEvaluation.printEvaluationMeasures(clusterId2Trajectory, trajectory2ClusterId)