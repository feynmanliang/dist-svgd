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

def get_results_dir(dataset_name, fold, nproc, nparticles, stepsize, exchange, wasserstein):
    subdir = 'logreg_{}_{}-nshards={}-nparticles={}-exchange={}-wasserstein={}-stepsize={:.0e}'.format(
            dataset_name, fold, nproc, nparticles, exchange, wasserstein, stepsize)
    return os.path.join(RESULTS_DIR, subdir)


def plot_test_acc(df, vis, plot_title, dataset_name, fold):
    # Load data
    mat = loadmat(os.path.join(DATA_DIR, 'benchmarks.mat'))
    dataset = mat[dataset_name][0, 0]

    # split #, instance, features/label
    x_train = torch.from_numpy(dataset[0][dataset[2] - 1][fold]).to(torch.float)
    t_train = dataset[1][dataset[2] - 1][fold]
    x_test = dataset[0][dataset[3] - 1][fold]
    t_test = dataset[1][dataset[3] - 1][fold]

    # Baseline test accuracy using sklearn
    baseline_test_acc = (LogisticRegression()
            .fit(x_train, t_train.reshape(-1))
            .score(x_test, t_test.reshape(-1)))

    # Ensemble test average
    def _test_acc(particles):
        "Computes test accuracy of posterior predictive mean over particles."
        def prob(particle):
            # Decode particle parameters
            alpha = np.exp(particle[0])
            w = particle[1:]
            return scipy.special.expit(x_test.dot(w))
        accuracy = ((particles.map(prob).mean() > 0.5).reshape(-1) == (t_test > 0).reshape(-1)).mean()
        return accuracy
    test_accs = (df
        .groupby('timestep', as_index=False)
        .apply(lambda x: pd.Series({
            'timestep': x['timestep'].max(),
            'dsvgd': _test_acc(x['value']),
            'sklearn logreg': baseline_test_acc,
        })))

    vis.line(
            Y=test_accs.drop('timestep', axis=1).values,
            X=test_accs['timestep'].values,
            opts=dict(
                xlabel='Iteration',
                ylabel='Test accuracy',
                title=plot_title,
                legend=test_accs.drop('timestep', axis=1).columns.values.tolist(),
                ))

def plot_w_scatters(df, vis, plot_title, timestep_between):
    # Particle positions
    for t in range(0, df['timestep'].max(), timestep_between):
        vis.scatter(
                X=np.stack(df[df['timestep'] == t]['value'].values)[:,1:3],
                opts=dict(
                    xlabel='w1',
                    xtickmin=-1.5,
                    xtickmax=1.5,
                    ylabel='w2',
                    ytickmin=-3,
                    ytickmax=2,
                    title=plot_title(t),
                    ))

def plot_alpha_hist(df, vis, plot_title, timestep_between):
    for t in range(0, df['timestep'].max(), timestep_between):
        vis.histogram(
                X=np.stack(df[df['timestep'] == t]['value'].values)[:,0],
                opts=dict(
                    xlabel='alpha',
                    xtickmin=-2,
                    xtickmax=2,
                    title=plot_title(t),
                    ))

@click.command()
@click.option('--dataset', type=click.Choice([
    'banana', 'diabetis', 'german', 'image', 'splice', 'titanic', 'waveform']), default='banana')
@click.option('--fold', type=int, default=42)
@click.option('--nproc', type=click.IntRange(0,32), default=1)
@click.option('--nparticles', type=int, default=10)
@click.option('--stepsize', type=float, default=1e-3)
@click.option('--exchange', type=click.Choice(['partitions', 'all_particles', 'all_scores']), default='partitions')
@click.option('--wasserstein/--no-wasserstein', default=False)
def make_plots(dataset, fold, nproc, nparticles, stepsize, exchange, wasserstein, **kwargs):
    # load run results
    results_dir = get_results_dir(dataset, fold, nproc, nparticles, stepsize, exchange, wasserstein)
    df = pd.concat(map(pd.read_pickle, glob(os.path.join(results_dir, 'shard-*.pkl'))))

    # Post-process and plot
    vis = visdom.Visdom()

    plot_title = 'logreg_{}_{} {} nshards={} nparticles={} exchange={} wasserstein={} stepsize={:.0e}'.format(
                    dataset, fold, 'test_acc', nproc, nparticles, exchange, wasserstein, stepsize)
    plot_test_acc(df, vis, plot_title, dataset, fold)

    if 'dataset' == 'banana':
        TIMESTEPS_BETWEEN_KDE_PLOTS = 10
        plot_title = lambda t: 'logreg_{}_{} {} t={} nshards={} nparticles={} exchange={} wasserstein={} stepsize={:.0e}'.format(
                        dataset, fold, 'particles_w1_w2', t, nproc, nparticles, exchange, wasserstein, stepsize)
        plot_w_scatters(df, vis, plot_title, TIMESTEPS_BETWEEN_KDE_PLOTS)

        plot_title = lambda t: 'logreg_{}_{} {} t={} nshards={} nparticles={} exchange={} wasserstein={} stepsize={:.0e}'.format(
                        dataset, fold, 'particles_alpha', t, nproc, nparticles, exchange, wasserstein, stepsize)
        plot_alpha_hist(df, vis, plot_title, TIMESTEPS_BETWEEN_KDE_PLOTS)

if __name__ == '__main__':
    make_plots()
