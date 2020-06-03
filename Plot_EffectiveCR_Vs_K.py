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
    k = pd['k']
    NADG = pd['NADG']
    EDG = pd['EDG']
    ODG = pd['ODG']
    CDG = pd['CDG']
    Best = pd['Best']
    return k,NADG,EDG,ODG,CDG,Best

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


    # plt.plot(x_plt, y_plt[0], color[0], marker=markers[0],markersize = 20, mfc='none')
    plt.show()


def plot1():

    y_plt = []
    k,NADG,EDG,ODG,CDG,Best = get_values("./OriginalData/EffectiveCR_Vs_K_(Obs0.3).csv")

    y_plt.append(NADG)
    y_plt.append(EDG)
    y_plt.append(ODG)
    y_plt.append(CDG)
    y_plt.append(Best)
    
    x_plt = np.arange(2,41,1)
    X_plt = np.arange(2,41,1)

    x_ticks = []
    for i in X_plt:
        if (i%5==0):
            x_ticks.append(i)
        else:
            x_ticks.append("")
	
    y_ticks = np.arange(0,2001,500)
    title = "EffectiveCR Vs Obstracles"
    x_label = "k"
    y_label = "CR"
    legends = ('NADG','EDG','ODG','CDG','Best')
    filename = "EffectiveCR_Vs_K.eps"
    
    plot_graph(x_plt, y_plt, X_plt, x_ticks, y_ticks, title, x_label, y_label, legends, [2,41], [0,2001], filename, 4)


if (__name__=='__main__'):
	plot1()