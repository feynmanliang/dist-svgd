import torch
from torch.distributions.normal import Normal

import pandas as pd

class Sampler(object):
    def __init__(self, d, logp, kernel):
        """Initializes a SVGD sampler.

        Params:
            d - dimensionality of each particle (e.g. number of parameters in Bayesian inference)
            kernel - kernel function
            logp - log likelihood function
        """
        self._d = d
        self._logp = logp
        self._kernel = kernel

    def _dkernel(self, x, y):
        "Returns \nabla_x k(x,y)."
        _x = x.detach()
        _x.requires_grad_(True)
        _y = y.detach()
        _y.requires_grad_(False)
        self._kernel(_x, _y).backward()
        return _x.grad

    def _dlogp(self, x):
        "Returns \nabla_x log p(x)"
        _x = x.detach()
        _x.requires_grad_(True)
        self._logp(_x).backward()
        return _x.grad

    def _phi_hat(self, particle, particles):
        total = torch.zeros(particle.size())
        for other_particle in particles:
            total += (self._kernel(other_particle, particle) * self._dlogp(other_particle)
                    + self._dkernel(other_particle, particle))
        return (1.0 / particles.shape[0]) * total

    def sample(self, n, num_iter, step_size):
        """Generate samples using SVGD.

        Params:
            n - number of particles (initialized using standard normals)
            num_iter - number of iterations of sampling to perform
            step_size - step size

        Returns:
            pandas dataframe with the following columps
                timestep - iteration step where sample in row was generated
                particle - integer identifier [0..n) of the particle
                value - values of this particle at this time
        """
        data = []

        q = Normal(0, 1)
        make_sample = lambda: q.sample((self._d, 1))
        particles = torch.cat([make_sample() for _ in range(n)], dim=1).t()

        for l in range(num_iter):
            print('Iteration {}'.format(l))
            for (i, particle) in enumerate(particles):
                # save results right before updating particles
                data.append(pd.Series([l, i, torch.tensor(particles[i]).numpy()], index=['timestep', 'particle', 'value']))

                particles[i] = particle + step_size * self._phi_hat(particle, particles)
            print(particles.mean(dim=0))

        # save results after last update
        for (i, particle) in enumerate(particles):
            data.append(pd.Series([l+1, i, torch.tensor(particles[i]).numpy()], index=['timestep', 'particle', 'value']))
        return pd.DataFrame(data)
