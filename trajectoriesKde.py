import random
import math

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

def plotTrajectories(trajDict):
    nrColors = len(trajDict)

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(nrColors)]

    colorsIterator = 0

    for trajId in trajDict:
        traj = trajDict[trajId]
        traj = np.array(traj)
        plt.plot(traj[:,0], traj[:,1], label = 'traj ' + str(trajId), color = colors[colorsIterator])
        colorsIterator += 1
    plt.show()

def kdePlotTrajs(trajDict):
    df = {'x': [], 'y': []}

    for trajId in trajDict:
        traj = trajDict[trajId]
        trajNp = np.array(traj)
        df['x'] += list(trajNp[:, 0])
        df['y'] += list(trajNp[:, 1])

    sns.kdeplot(x = df['x'], y = df['y'], shade=True)
    plt.show()

def convertToCartesian(elem):

    R = 1

    lon_r = elem[0]
    lat_r = elem[1]

    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    
    return (x, y)

def extractAngles(trajDict):

    trajectoryAngles = {}

    for trajId in trajDict:
        trajCartesian = [convertToCartesian((x, y)) for (x, y) in trajDict[trajId]]
        print(trajCartesian)
        trajectoryAngles[trajId] = [round(math.degrees(math.atan(y/x))) for (x, y) in trajCartesian]

    print(trajectoryAngles)

def basicStackOverflow():

    trajDict={}
    trajDict["traj1"] = [(0.0, 0.0), (0.00899, 0.26698), (-0.01028, 0.53258), (-0.0659, 0.79121), (-0.15139, 1.03582), (-0.25465, 1.27343), (-0.3676, 1.50964), (-0.48702, 1.73884), (-0.61451, 1.94978), (-0.75655, 2.13686)]
    trajDict["traj2"] = [(0.0, 0.0), (0.03813, 0.4454), (0.06093, 0.90909), (0.049, 1.35736), (0.01042, 1.80004), (-0.04754, 2.23151), (-0.11277, 2.64753), (-0.17074, 3.05652), (-0.22386, 3.4655), (-0.28748, 3.88152)]
    trajDict["traj3"] = [(0.0, 0.0), (0.02035, 0.39624), (0.01728, 0.7897), (-0.02375, 1.17196), (-0.10031, 1.54725), (-0.20029, 1.90851), (-0.30512, 2.24308), (-0.40755, 2.55095), (-0.49706, 2.84335), (-0.57284, 3.1287)]
    
    plotTrajectories(trajDict)

    

basicStackOverflow()