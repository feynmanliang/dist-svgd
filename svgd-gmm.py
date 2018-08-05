import math

import torch
from torch.distributions.normal import Normal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dsvgd.sampler

p1 = Normal(-2, 1)
p2 = Normal(2, 1)

def logp(x):
    # GMM mixture 1/3 * p1 + 2/3 * p2
    return torch.log(1. / 3. * torch.exp(p1.log_prob(x)) + 1. / 3. * torch.exp(p2.log_prob(x)))

def kernel(x, y):
    return torch.exp(-1.*torch.dist(x, y, p=2)**2)

n = 5
d = 1
num_iter = 5
step_size = 1e-0;

sampler = dsvgd.sampler.Sampler(d, logp, kernel)
df = sampler.sample(n, num_iter, step_size)
df['value'] = df['value'].map(lambda x: x[0])

sns.set()
g = sns.catplot(
        x='timestep',
        y='value',
        kind='violin',
        data=df)

g.set(xlim=(-12,8))

plt.show()

f, axes = plt.subplots(1, 6, figsize=(9,2))
for (i, timestep) in enumerate([0, 50, 75, 100, 150, 500]):
    sns.kdeplot(
            df[df['timestep'] == timestep].value,
            shade=True,
            legend=False,
            ax=axes[i])
    axes[i].set_title('Timestep {}'.format(timestep))

f.savefig('figures/gmm.png')
