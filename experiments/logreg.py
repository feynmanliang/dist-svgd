import os

from scipy.io import loadmat
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression

import torch
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from definitions import DATA_DIR, FIGURES_DIR
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

# Define model
d = 3

alpha_prior = Gamma(1, 1)
w_prior = lambda alpha: MultivariateNormal(torch.zeros(x_train.shape[1]), torch.eye(x_train.shape[1]) / alpha)

def logp(x):
    alpha = torch.exp(x[0])
    w = x[1:3].reshape((2,))
    logp = alpha_prior.log_prob(alpha)
    logp += w_prior(alpha).log_prob(w)
    logp += -torch.log(1. + torch.exp(-1.*torch.matmul(t_train * x_train, w))).sum()
    return logp

def kernel(x, y):
    return torch.exp(-1.*torch.dist(x, y, p=2)**2)

sampler = dsvgd.Sampler(d, logp, kernel)

# Define sampling parameters
n = 50
num_iter = 500
step_size = 3e-3

# Run sampler
df = sampler.sample(n, num_iter, step_size)

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
g.savefig(os.path.join(FIGURES_DIR, 'logreg-{}-test-acc.png'.format(dataset_name)))


g = sns.FacetGrid(df[df['timestep'] % 20 == 0], col="timestep")
def plot_kde(value, *args, **kwargs):
    ps = np.stack(value.values)[:,1:]
    ax = sns.kdeplot(ps[:,0],ps[:,1], *args, **kwargs)
    return ax
g.map(plot_kde, 'value')
g.savefig(os.path.join(FIGURES_DIR, 'logreg-{}-kde.png'.format(dataset_name)))
