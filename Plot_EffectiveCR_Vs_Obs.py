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
    Obs = pd['Obs']
    NADG = pd['EDG2']
    EDG = pd['EDG']
    ODG = pd['ODG']
    CDG = pd['CDG']
    return Obs,NADG,EDG,ODG,CDG
def plot_graph(x_plt, y_plt, X_plt, x_ticks, y_ticks, title, x_label, y_label, legends, xlim, ylim, filename, legend_pos):
    for i in range(len(y_plt)):
        plt.plot(x_plt, y_plt[i], color[i], marker=markers[i],markersize = 10, mfc='none')

    plt.legend(legends, loc = legend_pos, frameon=False, prop={'size': 20, 'weight':'normal'})
    plt.xticks(X_plt,x_ticks)
    plt.yticks(y_ticks)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)

    axes = plt.gca()
    axes.set_xlim([0.1,0.9])
    axes.set_ylim([0,1810])
    axes.set_xlabel(x_label, fontdict = font)
    axes.set_ylabel(y_label, fontdict = font)
    plt.tight_layout()
    plt.savefig(filename, format='eps', dpi=1000)


    # plt.plot(x_plt, y_plt[0], color[0], marker=markers[0],markersize = 20, mfc='none')
    plt.show()


def plot1():

    y_plt = []
    Obs,NADG,EDG,ODG,CDG = get_values("./OriginalData/EffectiveCR_Vs_Obs.csv")

    y_plt.append(NADG)
    y_plt.append(EDG)
    y_plt.append(ODG)
    y_plt.append(CDG)
    # print(y_plt)
    x_plt = np.arange(0.1,1,0.1)
    X_plt = np.arange(0.1,1,0.1)
    x_ticks = []
    for i in X_plt:
        # if (i%10==0):
        x_ticks.append(i)
        # else:
            # x_ticks.append("")
	
    y_ticks = np.arange(0,1801,200)
    title = "EffectiveCR Vs Obstracles"
    x_label = "Obs"
    y_label = "CR"
    legends = ('NADG','EDG','ODG','CDG')
    filename = "EffectiveCR_Vs_Obs.eps"
    
    plot_graph(x_plt, y_plt, X_plt, x_ticks, y_ticks, title, x_label, y_label, legends, [0.1,0.9], [0,1810], filename, 3)


if (__name__=='__main__'):
	plot1()