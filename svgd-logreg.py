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

mat = loadmat('data/benchmarks.mat')

dataset_name = 'banana'
dataset = mat[dataset_name][0, 0]

#for fold in x_train.shape[0]:
fold = 42

# split #, instance, features/label
x_train = torch.from_numpy(dataset[0][dataset[2] - 1][fold]).to(torch.float)
t_train = dataset[1][dataset[2] - 1][fold]
x_test = dataset[0][dataset[3] - 1][fold]
t_test = dataset[1][dataset[3] - 1][fold]

# TODO: train on x_train[fold], eval on x_test[fold]

alpha_prior = Gamma(1, 1)
w_prior = lambda alpha: MultivariateNormal(torch.zeros(x_train.shape[1]), torch.eye(x_train.shape[1]) / alpha)

def dlogp(x):
    _x = x.detach()
    _x.requires_grad_(True)
    alpha = torch.exp(_x[0])
    w = _x[1:].reshape((2,))
    logp = alpha_prior.log_prob(alpha)
    logp += w_prior(alpha).log_prob(w)
    logp += -torch.log(1. + torch.exp(-1.*torch.matmul(t_train * x_train, w))).sum()
    logp.backward()
    return _x.grad

def kernel(x, y):
    return torch.exp(-1.*torch.dist(x, y, p=2)**2)

def dkernel(x, y):
    "Returns \nabla_x k(x,y)."
    _x = x.detach()
    _x.requires_grad_(True)
    dkernel = kernel(_x, y)
    dkernel.backward()
    return _x.grad


n = 50
num_iter = 200
step_size = 1e-3

def phi_hat(particle, particles):
    total = torch.zeros(particle.size())
    for i,other_particle in enumerate(particles):
        total += kernel(other_particle, particle) * dlogp(other_particle) + dkernel(other_particle, particle)
    return (1.0 / n) * total

q = Normal(0, 1)
make_sample = lambda: torch.cat([q.sample((1,1)) - 0.5, q.sample((1,1)), q.sample((1,1))])
particles = torch.cat([make_sample() for _ in range(n)], dim=1).t()

data = []

for l in range(num_iter+1):
    print('Iteration {}'.format(l))
    for (i, particle) in enumerate(particles):
        particles[i] = particle + step_size * phi_hat(particle, particles)
        data.append(pd.Series([l, i, torch.tensor(particles[i]).numpy()], index=['timestep', 'particle', 'value']))


def test_acc(values):
    alpha = np.exp(values[0])
    w = values[1:]
    accuracy = ((x_test.dot(w) > 0).reshape(-1) == (t_test > 0).reshape(-1)).mean()
    return accuracy

df = pd.DataFrame(data)
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
g.savefig('figures/logreg-{}-test-acc.png'.format(dataset_name))


g = sns.FacetGrid(df[df['timestep'] % 20 == 0], col="timestep")
def plot_kde(value, *args, **kwargs):
    ps = np.stack(value.values)[:,1:]
    ax = sns.kdeplot(ps[:,0],ps[:,1], *args, **kwargs)
    return ax
g.map(plot_kde, 'value')
g.savefig('figures/logreg-{}-kde.png'.format(dataset_name))
