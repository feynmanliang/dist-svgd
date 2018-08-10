"""
Plots results.
"""
import os
from glob import glob

import click
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.io import loadmat
import scipy.special
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import torch

from definitions import RESULTS_DIR, DATA_DIR, FIGURES_DIR
import dsvgd

def make_plots(nproc, nparticles, stepsize, exchange, wasserstein):
    # Load data
    dataset_name = 'banana'
    mat = loadmat(os.path.join(DATA_DIR, 'benchmarks.mat'))
    dataset = mat[dataset_name][0, 0]
    fold = 42 # use 42 train/test split

    # split #, instance, features/label
    x_train = torch.from_numpy(dataset[0][dataset[2] - 1][fold]).to(torch.float)
    t_train = dataset[1][dataset[2] - 1][fold]
    x_test = dataset[0][dataset[3] - 1][fold]
    t_test = dataset[1][dataset[3] - 1][fold]

    # load run results
    df = pd.concat(map(pd.read_pickle, glob(os.path.join(RESULTS_DIR, 'shard-*.pkl'))))

    def save_fig(g, figname):
        g.fig.suptitle("nshards={}, nparticles={}, exchange={}, wasserstein={}, stepsize={:.0e}".format(
            nproc, nparticles, exchange, wasserstein, stepsize))
        g.savefig(os.path.join(
            FIGURES_DIR,
            'logreg-{}-{}-nproc={}-nparticles={}-stepsize={}-exchange={}-wasserstein={}.png'.format(
                dataset_name,
                figname,
                nproc,
                nparticles,
                stepsize,
                exchange,
                wasserstein)))

    # Post-process and plot
    sns.set()

    baseline_test_acc = (LogisticRegression()
            .fit(x_train, t_train.reshape(-1))
            .score(x_test, t_test.reshape(-1)))

    def test_acc(values):
        def prob(value):
            alpha = np.exp(value[0])
            w = value[1:]
            return scipy.special.expit(x_test.dot(w))
        accuracy = ((values.map(prob).mean() > 0.5).reshape(-1) == (t_test > 0).reshape(-1)).mean()
        return accuracy

    test_accs = (df
        .groupby('timestep', as_index=False)
        .apply(lambda x: pd.Series({
            'timestep': x['timestep'].max(),
            'dsvgd': test_acc(x['value']),
            'sklearn logreg': baseline_test_acc,
        }))
        .melt(id_vars=['timestep'], value_name='test acc'))

    g = sns.relplot(x='timestep', y='test acc', hue='variable', kind='line', data=test_accs)
    save_fig(g, 'testacc')

    g = sns.FacetGrid(df[df['timestep'] % 20 == 0], col="timestep")
    def plot_kde(value, *args, **kwargs):
        ps = np.stack(value.values)[:,1:]
        ax = sns.kdeplot(ps[:,0],ps[:,1], *args, **kwargs)
        return ax
    g.map(plot_kde, 'value')
    save_fig(g, 'kde')
