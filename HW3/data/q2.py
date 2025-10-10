import os
import sklearn.preprocessing as skpp
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.io as spio
from scipy import stats
from sklearn.neighbors import KernelDensity
from scipy.sparse.csgraph import shortest_path
import sklearn
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

def import_data():
    df = pd.read_csv("n90pol.csv")
    y = df['orientation']
    data = df.drop(['orientation'], axis=1)
    # data = sklearn.preprocessing.scale(data)
    return data, y

def calc_optimal_bins(data):
    # using Freedman-Diaconis from https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram
    n = len(data)
    iqr = stats.iqr(data)
    h = 2 * iqr / (n**(1/3))  # bin width
    range_data = np.max(data) - np.min(data)
    return int(np.ceil(range_data / h))

def calc_optimal_bw(data):
    # Silverman's formula from https://en.wikipedia.org/wiki/Kernel_density_estimation
    n = len(data)
    sigma = np.std(data)
    iqr = stats.iqr(data)
    return 0.9 * min(np.std(data), stats.iqr(data)/1.34) * n**(-0.2)

def plot_1d(data, title, bw, bins):
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.hist(data, bins = bins, density = True)
    ax1.set_title(f'{title} - Histogram ({bins} bins)')

    kde = KernelDensity(bandwidth = bw, kernel = 'gaussian').fit(data.reshape(-1,1))
    x_plot = np.linspace(np.min(data), np.max(data), 100).reshape(-1,1)
    density = kde.score_samples(x_plot)
    ax2.plot(x_plot, np.exp(density))
    ax2.set_title(f'{title} - KDE (bw = {bw:.3f})')

    plt.show()
    pass

def experiment_1d(data, title,bw,bins):
    bin_list = [int(.5*bins), bins, int(1.5*bins), int(2*bins)]
    bw_list = [.5*bw, bw, 1.5*bw, 2*bw]

    fig, axs = plt.subplots(4,2)

    for i, b in enumerate(bin_list):
        axs[i,0].hist(data, bins=b)
        axs[i,0].set_ylabel(f'({b} bins)',rotation=90,fontsize = 10)

    for i, b in enumerate(reversed(bw_list)):
        kde = KernelDensity(bandwidth = b, kernel = 'gaussian').fit(data.reshape(-1,1))
        x_plot = np.linspace(np.min(data), np.max(data), 100).reshape(-1,1)
        density = kde.score_samples(x_plot)
        axs[i,1].plot(x_plot, np.exp(density))
        axs[i,1].set_ylabel(f'(bw={b:.3f})',rotation=90,fontsize = 10)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"1D_{title}.png")
    pass

def experiment_1d_just_kdes(data1, data2,bw1, bw2):
    bw1_list = [.5*bw1, bw1, 1.5*bw1, 2*bw1]
    bw2_list = [.5*bw2, bw2, 1.5*bw2, 2*bw2]

    fig, axs = plt.subplots(4,2)
    axs[0,0].set_title('Amygdala')
    axs[0,1].set_title('ACC')

    for i, b in enumerate(reversed(bw1_list)):
        kde = KernelDensity(bandwidth = b, kernel = 'gaussian').fit(data1.reshape(-1,1))
        x_plot = np.linspace(np.min(data1), np.max(data1), 100).reshape(-1,1)
        density = kde.score_samples(x_plot)
        axs[i,0].plot(x_plot, np.exp(density))
        axs[i,0].set_ylabel(f'(bw={b:.3f})',rotation=90,fontsize = 10)
    for i, b in enumerate(reversed(bw2_list)):
        kde = KernelDensity(bandwidth = b, kernel = 'gaussian').fit(data2.reshape(-1,1))
        x_plot = np.linspace(np.min(data2), np.max(data2), 100).reshape(-1,1)
        density = kde.score_samples(x_plot)
        axs[i,1].plot(x_plot, np.exp(density))
        axs[i,1].set_ylabel(f'(bw={b:.3f})',rotation=90,fontsize = 10)
    title = f"1D OR{orientation} amyg+acc"
    fig.suptitle(f"OR:{orientation}")
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    pass

def plot_2d(data1, data2, title, bw1, bw2, bins1, bins2):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,12))
    
    # 2D Histogram
    ax1.hist2d(data1, data2, bins=[bins1,bins2])
    ax1.set_xlabel('Amygdala')
    ax1.set_ylabel('ACC')
    ax1.set_title(f'{title} - bins:{bins1}x{bins2}', fontsize=10)
    
    # 2D KDE
    silverman_kde = gaussian_kde(np.vstack([data1,data2]), bw_method='silverman')
    print(f"Auto-calculated Silverman bw: {silverman_kde.factor:.4f}")
    df = pd.DataFrame({'Amygdala': data1, 'ACC': data2})
    sns.kdeplot(data=df, 
                x='Amygdala', y='ACC',
                bw_method='silverman',
                fill=True,
                ax=ax2) 
    ax2.set_xlabel('Amygdala')
    ax2.set_ylabel('ACC')
    ax2.set_title(f'{title} auto bw', fontsize=10)

    # experimented with optimal BWs, but the composite works better. THis makes sense, since 1d optimal was also silverman method
    # 2D KDE w/ specific bws
    df = pd.DataFrame({'Amygdala': data1, 'ACC': data2})
    spec = sns.kdeplot(data=df, 
                x='Amygdala', y='ACC',
                fill=True,
                bw_method='silverman',
                bw_adjust=0.75,
                ax=ax3) 
    ax3.set_xlabel('Amygdala')
    ax3.set_ylabel('ACC')
    ax3.set_title(f'{title}.75 bw', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"2D_{title}.png")
    pass



if __name__ == "__main__":
    data, y = import_data()

    bw_acc = calc_optimal_bw(data['acc'].values)
    bins_acc = calc_optimal_bins(data['acc'].values)

    bw_amyg = calc_optimal_bw(data['amygdala'].values)
    bins_amyg = calc_optimal_bins(data['amygdala'].values)

    # plot_1d(data['amygdala'].values, 'Amygdala', bw_amyg, bins_amyg)
    experiment_1d(data['acc'].values, 'ACC', bw_acc, bins_acc)
    experiment_1d(data['amygdala'].values, 'Amyg', bw_amyg, bins_amyg)

    # comparing graphs and logging most suitable params
    # turned out to be the same as the optimal calculations
    best_bins_acc = bins_acc
    best_bins_amyg = bins_amyg
    best_bw_acc = bw_acc
    best_bw_amyg = bw_amyg

    plot_2d(data['amygdala'].values, data['acc'].values, '2D', best_bw_amyg, best_bw_acc, best_bins_amyg, best_bins_acc)
    
    # conditional 1d distributions
    for orientation in y.unique():
        or_filtered = data[y == orientation]
        bw_acc = calc_optimal_bw(or_filtered['acc'].values)
        bins_acc = calc_optimal_bins(or_filtered['acc'].values)
        bw_amyg = calc_optimal_bw(or_filtered['amygdala'].values)
        bins_amyg = calc_optimal_bins(or_filtered['amygdala'].values)

        # calc means
        mean_acc = np.mean(or_filtered['acc'].values)
        mean_amyg = np.mean(or_filtered['amygdala'].values)
        print(f"Means | O {orientation} | amyg {mean_amyg:.3f} | acc {mean_acc:.3f}")

        experiment_1d_just_kdes(or_filtered['amygdala'].values, or_filtered['acc'].values, bw_amyg, bw_acc)
        # the optimally calculated bins are serviceable and the bws are good

        plot_2d(or_filtered['amygdala'].values, or_filtered['acc'].values, f'2D OR{orientation}', bw_amyg, bw_acc, bins_amyg, bins_acc)

    # conditional 2d distributions
    pass