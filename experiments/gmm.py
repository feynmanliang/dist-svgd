import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.distributions.normal import Normal

from definitions import FIGURES_DIR
import dsvgd

torch.manual_seed(42)

# Define model
d = 1

p1 = Normal(-2, 1)
p2 = Normal(2, 1)

def logp(x):
    # GMM mixture 1/3 * p1 + 2/3 * p2
    return torch.log(1. / 3. * torch.exp(p1.log_prob(x)) + 1. / 3. * torch.exp(p2.log_prob(x)))

def kernel(x, y):
    return torch.exp(-1.*torch.dist(x, y, p=2)**2)

sampler = dsvgd.Sampler(d, logp, kernel)

# Define sampling parameters
n = 5
num_iter = 5
step_size = 1e-0;

# Run sampler
df = sampler.sample(n, num_iter, step_size)

# Post-process and plot
df['value'] = df['value'].map(lambda x: x[0])
sns.set()
f, axes = plt.subplots(1, 6, figsize=(9,2))
for (i, timestep) in enumerate([0, 50, 75, 100, 150, 500]):
    sns.kdeplot(
            df[df['timestep'] == timestep].value,
            shade=True,
            legend=False,
            ax=axes[i])
    axes[i].set_title('Timestep {}'.format(timestep))
f.savefig(os.path.join(FIGURES_DIR, 'gmm.png'))
