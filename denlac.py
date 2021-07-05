__author__ = "Radulescu Iulia-Maria"
__copyright__ = "Copyright 2017, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "iulia.radulescu@cs.pub.ro"
__status__ = "Production"

import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import matplotlib.pyplot as plt
import similaritymeasures
import utm
from random import randint
import argparse
import math
import collections
import time

class Denlac:

    def __init__(self, noClusters, noBins, expandFactor, noDims, aggMethod, debugMode):

        self.no_clusters = noClusters
        self.no_bins = noBins
        self.expandFactor = expandFactor  # expantion factor how much a cluster can expand based on the number of neighbours -- factorul cu care inmultesc closest mean (cat de mult se poate extinde un cluster pe baza vecinilor)

        self.noDims = noDims
        self.debugMode = debugMode
        self.aggMethod = aggMethod

        self.id_cluster = -1

    def rebuildDictIndexes(self, dictToRebuild, distBetweenPartitionsCache):

        newDict = dict()
        newCacheDict = dict()
        newDictIdx = 0

        oldNewIndexesCorrelation = {}

        for i in dictToRebuild:
            newDict[newDictIdx] = dictToRebuild[i]
            oldNewIndexesCorrelation[i] = newDictIdx
            newDictIdx = newDictIdx + 1

        for keyTuple in distBetweenPartitionsCache:
            if(keyTuple[0] in oldNewIndexesCorrelation and keyTuple[1] in oldNewIndexesCorrelation):
                newI = oldNewIndexesCorrelation[keyTuple[0]]
                newJ = oldNewIndexesCorrelation[keyTuple[1]]
                newCacheDict[(newI, newJ)] = distBetweenPartitionsCache[keyTuple]

        return newDict, newCacheDict, newDictIdx

    def computeDistanceIndices(self, partitions, distBetweenPartitionsCache):

        distances = []
        for i in partitions:
            for j in partitions:
                if (i == j):
                    distBetweenPartitions = -1
                else:
                    if (i, j) in distBetweenPartitionsCache:
                        distBetweenPartitions = distBetweenPartitionsCache[(i, j)]
                    else:
                        distBetweenPartitions = self.calculateSmallestPairwise(partitions[i], partitions[j])
                        distBetweenPartitionsCache[(i, j)] = distBetweenPartitions

                distances.append(distBetweenPartitions)

        # sort by distance
        distances = np.array(distances)
        indices = np.argsort(distances)

        finalIndices = [index for index in indices if distances[index] > 0]

        return finalIndices

    # i = index, x = amount of columns, y = amount of rows
    def indexToCoords(self, index, columns, rows):

        for i in range(rows):
            # check if the index parameter is in the row
            if (index >= columns * i and index < (columns * i) + columns):
                # return x, y
                return index - columns * i, i

    def sortAndDeduplicate(self, l):

        result = []
        for value in l:
            if any(np.array_equal(value, x) for x in result) == False:
                result.append(value)

        return result

    def joinPartitions(self, final_partitions, finalNoClusters):

        partitions = dict()
        partId = 0

        for k in final_partitions:
            partitions[partId] = []
            partId = partId + 1

        partId = 0

        for k in final_partitions:
            for point in final_partitions[k]:
                partitions[partId].append(point[0])
            partId = partId + 1

        distBetweenPartitionsCache = {}
        distancesIndices = self.computeDistanceIndices(partitions, distBetweenPartitionsCache)

        while len(partitions) > finalNoClusters:

            smallestDistancesIndex = distancesIndices[0]

            (j, i) = self.indexToCoords(smallestDistancesIndex, len(partitions), len(partitions))
            partitionToAdd = partitions[i] + partitions[j]
            partitionToAdd = self.sortAndDeduplicate(partitionToAdd)

            if (i in partitions):
                del partitions[i]

            if (j in partitions):
                del partitions[j]

            if ((i,j) in distBetweenPartitionsCache):
                del distBetweenPartitionsCache[(i,j)]

            (partitions, distBetweenPartitionsCache, newDictIdx) = self.rebuildDictIndexes(partitions, distBetweenPartitionsCache)

            partitions[newDictIdx] = partitionToAdd

            distancesIndices = self.computeDistanceIndices(partitions, distBetweenPartitionsCache)

        return partitions
        
    '''
        compute pdf and its values for points in dataset
    '''
    def computePdfKde(self, dataset):

        # prepare kde dataset list
        kdeList = []

        for dim in range(2):
            q = dataset[:, :, dim]
            q = q.reshape(np.shape(q)[0]*np.shape(q)[1], 1)
            kdeList.append(q)
        
        kde = KDEMultivariate(data=kdeList, var_type='cc', bw='normal_reference')
        pdf = kde.pdf(kdeList)
        return pdf

    def DistFunc(self, x, y):

        return similaritymeasures.frechet_dist(x, y)

    def DistFuncMeanHaversine(self, x, y):
        hDistanceMatrix = haversine_distances(x, y) * 3959
        distances = []
        for i in range(np.shape(hDistanceMatrix)[0]):
            for j in range(np.shape(hDistanceMatrix)[1]):
                if (j < i):
                    continue
                distances.append(hDistanceMatrix[i][j])  
        return np.mean(np.array(distances))

    def outliersIqr(self, ys):
        '''
		Outliers detection with IQR
		'''
        quartile1, quartile3 = np.percentile(ys, [25, 75])
        iqr = quartile3 - quartile1
        lowerBound = quartile1 - (iqr * 1.5)
        outliersIqrIds = []
        for idx in range(len(ys)):
            if ys[idx] < lowerBound:
                outliersIqrIds.append(idx)
        return outliersIqrIds

    def calculateSmallestPairwise(self, cluster1, cluster2):

        minPairwise = 999999
        for traj1 in cluster1:
            for traj2 in cluster2:
                comparison = traj1 == traj2
                if (comparison.all() == False):
                    distBetween = self.DistFunc(traj1, traj2)
                    if (distBetween < minPairwise):
                        minPairwise = distBetween
        return minPairwise


    def computeDistanceMatrix(self, trajectories):

        nrPoints = len(trajectories)

        # compute distance matrix
        distanceMatrix = np.zeros((nrPoints, nrPoints))

        for trajId1 in range(nrPoints):
            for trajId2 in range(nrPoints):
                distanceMatrix[trajId1][trajId2] = self.DistFunc(trajectories[trajId1], trajectories[trajId2])

        return distanceMatrix

    
    def getDistancesToKthNeigh(self, kthNeigh, trajectories):

        distanceMatrix = self.computeDistanceMatrix(trajectories)

        # compute distances to closest K neigh
        distancesToClosestK = distanceMatrix[:, kthNeigh]

        # return k distances, sorted descending
        return np.array(sorted(distancesToClosestK, reverse=True))

    def getCorrectRadius(self, pointsPartition):

        ns = 3

        # distances to the kth nearest neighbor, sorted
        distanceDec = self.getDistancesToKthNeigh(ns, [point[0] for point in pointsPartition])

        print(distanceDec)

        maxSlopeIdx = np.argmax(distanceDec[:-1] - distanceDec[1:])

        # plt.plot(distanceDec)
        # plt.axvline(x=distanceDec[maxSlopeIdx], color='k', label=f'Inflection Point')
        # plt.show()

        return distanceDec[maxSlopeIdx]


    def getClosestKNeigh(self, idPoint, pointsPartition, closestMean):
        '''
		Get a point's closest v neighbours
		v is not a constant!! for each point you keep adding neighbours
		untill the distance from the next neigbour and the point is larger than
		expand_factor * closestMean (closestMean este calculata de functia anterioara)
		'''
        radius = self.expandFactor * closestMean

        neighIdsToDistances = {}

        for idPoint2 in range(len(pointsPartition)):
            if (idPoint == idPoint2):
                continue
            neighIdsToDistances[idPoint2] = self.DistFunc(pointsPartition[idPoint][0], pointsPartition[idPoint2][0])

        closestKNeigh = []

        for key, distance in neighIdsToDistances.items():
            if distance <= radius:
                closestKNeigh.append(key)

        return closestKNeigh

    def expandKnn(self, pointId, pointsPartition, closestMean):
        '''
		Extend current cluster
		Take the current point's nearest v neighbours
		Add them to the cluster
		Take the v neighbours of the v neighbours and add them to the cluster
		When you can't expand anymore start new cluster
		'''

        neighIds = self.getClosestKNeigh(pointId, pointsPartition, closestMean)

        if (len(neighIds) > 0):
            pointsPartition[pointId][1] = self.id_cluster
            pointsPartition[pointId][3] = 1
            
            for neighId in neighIds:
                if (pointsPartition[neighId][1] == -1):
                    self.expandKnn(neighId, pointsPartition, closestMean)
        else:
            pointsPartition[pointId][1] = -1
            pointsPartition[pointId][3] = 1

    def splitPartitions(self, partitionDict):

        print("Expand factor " + str(self.expandFactor))
        noise = []
        noClustersPartition = 1
        partId = 0
        finalPartitions = collections.defaultdict(list)

        for k in partitionDict: # for each bin, where k is the binId

            # EXPANSION STEP
            self.id_cluster = -1
            pointsPartition = partitionDict[k] # get trajectories in that bin

            closestMean = self.getCorrectRadius(pointsPartition)

            print('closestMean', closestMean)

            for pointId in range(len(pointsPartition)):

                pointActualValues = pointsPartition[pointId][0]

                if (pointsPartition[pointId][1] == -1):
                    self.id_cluster = self.id_cluster + 1
                    noClustersPartition = noClustersPartition + 1
                    pointsPartition[pointId][3] = 1
                    pointsPartition[pointId][1] = self.id_cluster
                    neigh_ids = self.getClosestKNeigh(pointId, pointsPartition, closestMean)

                    for neigh_id in neigh_ids:
                        if (pointsPartition[neigh_id][1] == -1):
                            pointsPartition[neigh_id][3] = 1
                            pointsPartition[neigh_id][1] = self.id_cluster
                            self.expandKnn(neigh_id, pointsPartition, closestMean)

            # ARRANGE STEP
            # create partitions
            innerPartitions = collections.defaultdict(list)
            partIdInner = 0
            for i in range(noClustersPartition):
                innerPartitions[partIdInner] = [pointActualValues for pointActualValues in pointsPartition if pointActualValues[1] == i]
                partIdInner = partIdInner + 1

            noise += [pointActualValues for pointActualValues in pointsPartition if pointActualValues[1] == -1]

            # filter partitions - eliminate the ones with a single point and add them to the noise list
            keysToDelete = []
            for k in innerPartitions:
                if (len(innerPartitions[k]) <= 1):
                    keysToDelete.append(k)
                    # we save these points and assign them to the closest cluster
                    if (len(innerPartitions[k]) > 0):
                        noise += [pointActualValues for pointActualValues in innerPartitions[k]]

            for k in keysToDelete:
                del innerPartitions[k]

            # reindex dict
            innerPartitionsFiltered = dict(zip(range(0, len(innerPartitions)), list(innerPartitions.values())))

            for partIdInner in innerPartitionsFiltered:
                finalPartitions[partId] = innerPartitionsFiltered[partIdInner]
                partId = partId + 1

        return (finalPartitions, noise)

    def addNoiseToFinalPartitions(self, noise, joinedPartitions):
        noise_to_partition = collections.defaultdict(list)
        # reassign the noise to the class that contains the nearest neighbor
        for noise_point in noise:
            # determine which is the closest cluster to noise_point
            closest_partition_idx = 0
            minDist = 99999
            for k in joinedPartitions:
                dist = self.calculateSmallestPairwise([noise_point[0]], joinedPartitions[k])
                if (dist < minDist):
                    closest_partition_idx = k
                    minDist = dist
            noise_to_partition[closest_partition_idx].append(noise_point[0])

        for joinedPartId in noise_to_partition:
            for noise_point in noise_to_partition[joinedPartId]:
                joinedPartitions[joinedPartId].append(noise_point)


    def clusterDataset(self, dataset, evaluatePerf = False):

        intermediaryPartitionsDict = collections.defaultdict(list)

        pdf = self.computePdfKde(dataset)  # compute pdf using kde with custom dist (frechet)

        '''
        Detect and eliminate outliers
        '''
        outliersIqrPdf = self.outliersIqr(pdf)
        print("We identified " + str(len(outliersIqrPdf)) + " outliers from " + str(len(dataset)) + " points")

        # recompute dataset
        filterIndices = [q for q in range(len(dataset)) if q not in outliersIqrPdf]
        dataset = np.take(dataset, filterIndices, 0)

        '''
         Compute dataset pdf
        '''
        pdf = self.computePdfKde(dataset)  # calculez functia densitate probabilitate din nou

        '''
		Split the dataset in density bins
		'''
        _, bins = np.histogram(pdf, bins=self.no_bins)

        for idxBin in range((len(bins) - 1)):
            for idxPoint in range(len(dataset)):
                if (pdf[idxPoint] >= bins[idxBin] and pdf[idxPoint] <= bins[idxBin + 1]):
                    element_to_append = []
                    element_to_append.append(dataset[idxPoint])
                    # additional helpful values
                    element_to_append.append(-1)  # the split nearest-neighbour cluster the point belongs to
                    element_to_append.append(pdf[idxPoint])
                    element_to_append.append(-1)  # was the point already parsed?

                    intermediaryPartitionsDict[idxBin].append(element_to_append)

        '''
		Density levels bins distance split
		'''
        final_partitions, noise = self.splitPartitions(intermediaryPartitionsDict)  # functie care scindeaza partitiile

        print('noise points ' + str(len(noise)) + ' from ' + str(len(dataset)) + ' points')

        '''
        Joining partitions based on distances
         '''
        joinedPartitions = self.joinPartitions(final_partitions, self.no_clusters)

        '''
        Adding what was classified as noise to the corresponding partition
        '''
        self.addNoiseToFinalPartitions(noise, joinedPartitions)

        return joinedPartitions

'''
the dataset consists of n trajectories, each with 10 points, each point with 2 dimensions
'''
def runDenLAC(dataset):

    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-nclusters', '--nclusters', type = int, help = "the desired number of clusters")
    parser.add_argument('-nbins', '--nbins', type = int, help = "the number of density levels of the dataset")
    parser.add_argument('-expFactor', '--expansionFactor', type = float, help = "between 0.2 and 1.5 - the level of wideness of the density bins")
    parser.add_argument('-aggMethod', '--agglomerationMethod', type=int,
                        help="1 smallest pairwise (default) or 2 centroidclo", default = 1)
    parser.add_argument('-dm', '--debugMode', type = int,
                        help = "optional, set to 1 to show debug plots and comments for 2 dimensional datasets", default = 0)
    args = parser.parse_args()

    no_clusters = int(args.nclusters)  # no clusters
    no_bins = int(args.nbins)  # no bins
    expand_factor = float(args.expansionFactor)  # expansion factor how much a cluster can expand based on the number of neighbours -- factorul cu care inmultesc closest mean (cat de mult se poate extinde un cluster pe baza vecinilor)
    aggMethod = int(args.agglomerationMethod)
    debugMode = args.debugMode

    noDims = np.shape(dataset)[1]

    denlacInstance = Denlac(no_clusters, no_bins, expand_factor, noDims, aggMethod, debugMode)
    joinedPartitions = denlacInstance.clusterDataset(dataset)

    end = time.time()
    print('It took ' + str(end - start))

    return joinedPartitions