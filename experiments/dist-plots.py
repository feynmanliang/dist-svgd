"""
Run with `python -m torch.distributed.launch --nproc_per_node=2 experiments/dist.py`
"""
import os
from glob import glob

from scipy.io import loadmat
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression

import torch

from definitions import RESULTS_DIR, DATA_DIR, FIGURES_DIR
import dsvgd

torch.manual_seed(42)

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

df = pd.concat(map(pd.read_pickle, glob(os.path.join(RESULTS_DIR, 'shard-*.pkl'))))

# Post-process and plot
sns.set()
def test_acc(values):
    alpha = np.exp(values[0])
    w = values[1:]
    accuracy = ((x_test.dot(w) > 0).reshape(-1) == (t_test > 0).reshape(-1)).mean()
    return accuracy

test_accs = (df
    .groupby('timestep', as_index=False)
    .apply(lambda x: pd.Series({
        'timestep': x['timestep'].max(),
        'mean_test_acc': x['value'].map(test_acc).mean(),
        'max_test_acc': x['value'].map(test_acc).max()
    }))
    .melt(id_vars=['timestep']))

baseline_test_acc = (LogisticRegression()
        .fit(x_train, t_train.reshape(-1))
        .score(x_test, t_test.reshape(-1)))

g = sns.relplot(x='timestep', y='value', hue='variable', kind='line', data=test_accs)
plt.axhline(baseline_test_acc, color='r')
g.savefig(os.path.join(FIGURES_DIR, 'logreg-dist-{}-test-acc.png'.format(dataset_name)))


g = sns.FacetGrid(df[df['timestep'] % 20 == 0], col="timestep")
def plot_kde(value, *args, **kwargs):
    ps = np.stack(value.values)[:,1:]
    ax = sns.kdeplot(ps[:,0],ps[:,1], *args, **kwargs)
    return ax
g.map(plot_kde, 'value')
g.savefig(os.path.join(FIGURES_DIR, 'logreg-dist-{}-kde.png'.format(dataset_name)))
