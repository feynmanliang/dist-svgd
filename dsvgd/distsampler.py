import torch
from torch.distributions.normal import Normal

import numpy as np
import scipy.optimize

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

    def _wasserstein_grad(self, particles, previous_particles):
        """Computes the gradient of the W2 distance.

        TODO: extend to when particles are not equal"""
        # solve the wasserstein LP (should be a matching)
        m = particles.shape[0]
        n = previous_particles.shape[0]
        d = particles[0].shape[0]
        diffs = np.zeros((m, n, d))
        for i in range(m):
            for j in range(n):
                diffs[i][j] = particles[i] - previous_particles[j]
        c = np.apply_along_axis(lambda x: np.linalg.norm(x, ord=2)**2, 2, diffs).flatten(order='C')
        A_eq = np.zeros((m+n, m*n))
        for i in range(n):
            A_eq[i,m*i:m*(i+1)] = 1
        for j in range(m):
            for k in range(n):
                A_eq[n+j, j + k*m] = 1
        b_eq = np.hstack([
            [ 1. / m for _ in range(m) ],
            [ 1. / n for _ in range(n) ]
        ]).squeeze()

        transport_plan = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq).x.reshape(n, m)

        return np.sum(np.expand_dims(transport_plan, axis=2) * diffs, axis=1)

    def make_step(self, particles_to_update, step_size, h=1.0, interacting_particles=None, previous_particles=None):
        """Performs one step of SVGD.

        Params:
            particles_to_update - particles to update (mutated in-place)
            step_size - step size
            h - discretization size for JKO scheme
            interacting_particles - particles to compute interactions with, should include `particles`
            previous_particles - particles to compute wasserstein distance gradient against

        Returns:
            reference to `particles`
        """

        if interacting_particles is None:
            interacting_particles = particles_to_update

        wasserstein_grad = None
        if previous_particles is not None:
            wasserstein_grad = self._wasserstein_grad(particles_to_update, previous_particles)
        for (i, particle) in enumerate(particles_to_update):
            delta = self._phi_hat(particle, interacting_particles)

            if wasserstein_grad is not None:
                delta += h * torch.from_numpy(wasserstein_grad[i,:]).float()

            particles_to_update[i] = particle + step_size * delta

        return particles_to_update
