import math
import torch

from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution

p1 = Normal(-2, 1)
p2 = Normal(2, 1)

def dlogp(x):
    _x = x.detach()
    _x.requires_grad_(True)
    # GMM mixture 1/3 * p1 + 2/3 * p2
    logp = torch.log(1. / 3. * torch.exp(p1.log_prob(_x)) + 1. / 3. * torch.exp(p2.log_prob(_x)))
    # logp = p2.log_prob(_x)
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

q = Normal(-10, 1)

n = 10
num_iter = 200
step_size = 1e-0;

def phi_hat(particle, particles):
    total = torch.zeros(particle.size())
    for other_particle in particles:
        total += kernel(other_particle, particle) * dlogp(other_particle) + dkernel(other_particle, particle)
    return (1.0 / n) * total

particles = q.sample(torch.Size([n, 1]))

for l in range(num_iter):
    print('iter ' + str(l))
    for (i, particle) in enumerate(particles):
        particles[i] = particle + step_size * phi_hat(particle, particles)
    print(particles.mean())
