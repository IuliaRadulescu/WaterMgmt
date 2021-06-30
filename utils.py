import pandas as pd
import os

def txtToCSV(filename):

    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    fileLocation = scriptDirectory + '/trajectories/czech_june_2021/'

    trajFileHandler = open(fileLocation + filename, 'r')

    csvLines = 'ntra,2,3,4,5,6,7,8,9,lat,lon,12,13,14,15,16,label\n'

    for line in trajFileHandler:
        line = ','.join(list(filter(lambda x: x != '', list(map(lambda x: x.strip(' \n'), line.split(' ')))))) + ',1'
        csvLines += line + '\n'

    trajFileHandler.close()
    trajCsvFileHandler = open(fileLocation + 'trajs.csv', 'w')

    trajCsvFileHandler.write(csvLines)


def readTraj():

    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    fileLocation = scriptDirectory + '/trajectories/czech_june_2021/trajs.csv'

    return pd.read_csv(fileLocation, sep=',')