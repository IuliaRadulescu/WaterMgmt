import os
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import utils


def plotTraj(trajDf, label='label'):

    _, ax = plt.subplots(figsize=(16,12))

    colors = np.tile(list(matplotlib.colors.TABLEAU_COLORS.values()), 100)

    lal = -10 # latitude low boundery
    lah = 60 # latitude high boundery
    lol = -40 # longitude low boundery
    loh = 60 # longitude high boundery

    world = gp.read_file(gp.datasets.get_path('naturalearth_lowres'))
    world.plot(ax=ax, color='white', edgecolor='black')

    ax.set_xlim(lol,loh)
    ax.set_ylim(lal,lah)


    grouped = trajDf.groupby(label)

    for name, group in grouped:
        #print(name)
        if name == -1:
            continue
        for _, g in group.groupby('ntra'):
            ax.plot(g.lon, g.lat, alpha=0.7, c=colors[name])

    ax.set_xlabel('Longitude [\N{DEGREE SIGN}]', fontsize=35)
    ax.set_ylabel('Latitude [\N{DEGREE SIGN}]', fontsize=35)

    ax.tick_params(labelsize=30)

    plt.show()

# utils.txtToCSV('endpoint_values.txt')
# trajDf = utils.readTraj()
# plotTraj(trajDf)