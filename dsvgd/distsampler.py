import torch
import torch.distributed as dist
from torch.distributions.normal import Normal

import numpy as np
import scipy.optimize

class DistSampler(object):
    def __init__(self, rank, num_shards, d, logp, kernel, particles):
        """Initializes a distributed SVGD sampler.

        Params:
            rank - rank of shard
            num_shards - total number of shards
            d - dimensionality of each particle (e.g. number of parameters in Bayesian inference)
            kernel - kernel function
            logp - log likelihood function
            particles - array containing local state of all the particles
        """
        self._rank = rank
        self._num_shards = num_shards
        self._d = d
        self._logp = logp
        self._kernel = kernel
        self._particles = particles
        self._previous_particles = None

        # NOTE: this will drop particles if not divisible by num_shards
        self._particles_per_shard = int(self._particles.shape[0] / self._num_shards)
        (start, end) = self._particle_idx_range(rank)
        self._particle_start_idx = start
        self._particle_end_idx = end

    @property
    def particles(self):
        "Returns particles currently being updated on this sampler"
        return self._particles[self._particle_start_idx:self._particle_end_idx,:]

    @particles.setter
    def particles(self, value):
        "Sets value of particles currently being updated on this sampler"
        assert value.shape == self.particles.shape
        self._particles[self._particle_start_idx:self._particle_end_idx,:] = value

    def _particle_idx_range(self, rank):
        assert rank >= 0 and rank < self._num_shards
        return (self._particles_per_shard * rank, self._particles_per_shard * (rank+1))

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

    def exchange_round_robin(self):
        "Exchanges single particle partitions round robin."
        send_to_rank = (self._rank + 1) % self._num_shards

        particles_to_send = torch.tensor(self.particles).contiguous()
        req = dist.isend(tensor=particles_to_send, dst=send_to_rank)

        # receive new particles into indices owned by other shard
        recv_from_rank = (self._rank - 1 + self._num_shards) % self._num_shards
        start, end = self._particle_idx_range(recv_from_rank)

        new_particles = torch.empty_like(self._particles[start:end,:])
        req2 = dist.irecv(tensor=new_particles, src=recv_from_rank)

        req.wait()
        req2.wait()

        self._particles[start:end,:] = new_particles
        self._particle_start_idx = start
        self._particle_end_idx = end

    def exchange_all(self):
        "Gathers all particles to all shards."
        tensor_list = [torch.empty(self._particles_per_shard, self._d) for _ in range(self._num_shards)]
        dist.all_gather(tensor_list, self.particles)
        self._particles = torch.cat(tensor_list)


    def make_step(self, step_size, h=1.0, include_wasserstein=False):
        """Performs one step of SVGD.

        Params:
            step_size - step size
            h - discretization size for JKO scheme
            include_wasserstein - whether to include a wasserstein term

        Returns:
            reference to `particles`
        """
        interacting_particles = self.particles

        wasserstein_grad = None
        if include_wasserstein and self._previous_particles is not None:
            wasserstein_grad = self._wasserstein_grad(self.particles, self._previous_particles)
        for (i, particle) in enumerate(self.particles):
            delta = self._phi_hat(particle, interacting_particles)

            if wasserstein_grad is not None:
                delta += h * torch.from_numpy(wasserstein_grad[i,:]).float()

            self.particles[i] += step_size * delta

        self._previous_particles = torch.tensor(self.particles)
