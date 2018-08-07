import torch
from torch.distributions.normal import Normal

import pandas as pd

class DistSampler(object):
    def __init__(self, d, logp, kernel):
        """Initializes a distributed SVGD sampler.

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

    def make_step(self, particles_to_update, interacting_particles, step_size):
        """Performs one step of SVGD.

        Params:
            particles_to_update - particles to update (mutated in-place)
            interacting_particles - particles to compute interactions with, should include `particles`
            step_size - step size

        Returns:
            reference to `particles`
        """
        for (i, particle) in enumerate(particles_to_update):
            particles_to_update[i] = particle + step_size * self._phi_hat(particle, interacting_particles)

        return particles_to_update
