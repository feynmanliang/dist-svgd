"""
Plots results.
"""
import os
from glob import glob

import click
import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.special
from sklearn.linear_model import LogisticRegression
import torch
import visdom

from definitions import RESULTS_DIR, DATA_DIR
import dsvgd

@click.command()
@click.option('--nproc', type=click.IntRange(0,32), default=1)
@click.option('--nparticles', type=int, default=10)
@click.option('--stepsize', type=float, default=1e-3)
@click.option('--exchange', type=click.Choice(['partitions', 'all_particles', 'all_scores']), default='partitions')
@click.option('--wasserstein/--no-wasserstein', default=False)
def make_plots(nproc, nparticles, stepsize, exchange, wasserstein, **kwargs):
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

    # Post-process and plot
    vis = visdom.Visdom()

    # Baseline test accuracy
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
        })))
    vis.line(
            Y=test_accs.drop('timestep', axis=1).values,
            X=test_accs['timestep'].values,
            opts=dict(
                xlabel='Iteration',
                ylabel='Test accuracy',
                title='logreg {} {} nshards={} nparticles={} exchange={} wasserstein={} stepsize={:.0e}'.format(
                    dataset_name, 'test_acc', nproc, nparticles, exchange, wasserstein, stepsize),
                legend=test_accs.drop('timestep', axis=1).columns.values.tolist(),
                ))

    # Particle positions
    for t in range(0, df['timestep'].max(), 10):
        vis.scatter(
                X=np.stack(df[df['timestep'] == t]['value'].values)[:,1:],
                opts=dict(
                    xlabel='w1',
                    xtickmin='-1.5',
                    xtickmax='1.5',
                    ylabel='w2',
                    ytickmin='-3',
                    ytickmax='2',
                    title='logreg {} {} t={} nshards={} nparticles={} exchange={} wasserstein={} stepsize={:.0e}'.format(
                        dataset_name, 'particles', t, nproc, nparticles, exchange, wasserstein, stepsize),
                    ))

if __name__ == '__main__':
    make_plots()
