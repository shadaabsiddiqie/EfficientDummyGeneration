import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
import pandas
import matplotlib.gridspec as gridspec

color = ['r','g','b','m','c','y','k','w']
markers = ['o','x','2','4','3','1']
font = {'weight': 'normal','size': 20,}
rc('axes', linewidth=2)

def get_values(file1):
    pd = pandas.read_csv(file1)
    k = pd['Rmin']
    # Rmin,NADG,EDG,Best
    NADG = pd['NADG']
    EDG = pd['EDG']
    Best = pd['Best']
    return k,NADG,EDG,Best

def plot_graph(x_plt, y_plt, X_plt, x_ticks, y_ticks, title, x_label, y_label, legends, xlim, ylim, filename, legend_pos):
    for i in range(len(y_plt)):
        plt.plot(x_plt, y_plt[i], color[i], marker=markers[i],markersize = 10, mfc='none')

    plt.legend(legends, loc = legend_pos,ncol = 2, frameon=False, prop={'size': 20, 'weight':'normal'})
    plt.xticks(X_plt,x_ticks)
    plt.yticks(y_ticks)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)

    axes = plt.gca()
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_xlabel(x_label, fontdict = font)
    axes.set_ylabel(y_label, fontdict = font)
    plt.tight_layout()
    plt.savefig(filename, format='eps', dpi=1000)
    plt.show()


def plot1():

    y_plt = []
    Rmin,NADG,EDG,Best = get_values("./OriginalData/EffectiveCR_Vs_Dmin.csv")

    y_plt.append(NADG)
    y_plt.append(EDG)
    y_plt.append(Best)
    
    x_plt = np.arange(1,26,1)
    X_plt = np.arange(1,26,1)

    x_ticks = []
    for i in X_plt:
        if (i%5==0):
            x_ticks.append(i)
        else:
            x_ticks.append("")
	
    y_ticks = np.arange(0,2001,200)
    title = "EffectiveCR Vs Rmin"
    x_label = "Rmin"
    y_label = "EffectiveCR"
    legends = ('NADG','EDG','Best')
    filename = "EffectiveCR_Vs_Dmin.eps"
    
    plot_graph(x_plt, y_plt, X_plt, x_ticks, y_ticks, title, x_label, y_label, legends, [1,26], [0,2000], filename, 4)


if (__name__=='__main__'):
	plot1()