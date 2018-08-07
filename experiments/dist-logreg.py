"""
Run with `python -m torch.distributed.launch --nproc_per_node=2 experiments/dist.py`
"""
import os

from scipy.io import loadmat
import pandas as pd

import torch
import torch.distributed as dist
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.multiprocessing import Process

from definitions import DATA_DIR, FIGURES_DIR, RESULTS_DIR
import dsvgd


def run():
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

    num_shards = dist.get_world_size()
    samples_per_shard = int(x_train.shape[0] / num_shards)

    # Define model
    d = 3

    alpha_prior = Gamma(1, 1)
    w_prior = lambda alpha: MultivariateNormal(torch.zeros(x_train.shape[1]), torch.eye(x_train.shape[1]) / alpha)

    # TODO: make these imbalanced
    def data_idx_range(rank):
        "Returns the (start,end) indices of the range of data belonging to worker with rank `rank`"
        return (samples_per_shard * rank, samples_per_shard * (rank+1))

    def logp(shard, x):
        # Get shard-local data
        shard_start_idx, shard_end_idx = data_idx_range(shard)
        x_train_local = x_train[shard_start_idx:shard_end_idx]
        t_train_local = t_train[shard_start_idx:shard_end_idx]

        alpha = torch.exp(x[0])
        w = x[1:3].reshape((2,))
        logp = alpha_prior.log_prob(alpha)
        logp += w_prior(alpha).log_prob(w)
        logp += -torch.log(1. + torch.exp(-1.*torch.matmul(t_train_local * x_train_local, w))).sum()
        return logp

    def kernel(x, y):
        return torch.exp(-1.*torch.dist(x, y, p=2)**2)

    rank = dist.get_rank()

    torch.manual_seed(rank)
    dist_sampler = dsvgd.DistSampler(d, (lambda x: logp(rank, x)), kernel)

    # Define sampling parameters
    n = 50
    num_iter = 500
    step_size = 3e-3

    def particle_idx_range(rank):
        particles_per_shard = int(n / num_shards)
        return (particles_per_shard * rank, particles_per_shard * (rank+1))
    particle_start_idx, particle_end_idx = particle_idx_range(rank)

    # Run sampler
    q = Normal(0, 1)
    make_sample = lambda: q.sample((d, 1))
    particles = torch.cat([make_sample() for _ in range(n)], dim=1).t()

    # particles currently "owned" by this shard (i.e. exchanged at end of iteration)
    particle_start_idx, particle_end_idx = particle_idx_range(rank)

    data = []

    for l in range(num_iter):
        if rank == 0:
            print('Iteration {}'.format(l))

        # save results right before updating particles
        for i in range(particle_start_idx, particle_end_idx):
            data.append(pd.Series([l, i, torch.tensor(particles[i]).numpy()], index=['timestep', 'particle', 'value']))

        # Only update "owned" particles
        # particles_to_update = particles[particle_start_idx:particle_end_idx,:]

        # Update all local particles, regardless of "owned" or not
        # NOTE: this makes the algorithm much slower, lose parallelism via sharding particles
        particles_to_update = particles

        # Interact only with particles assigned to this worker at this iteration
        # interacting_particles = particles_to_update

        # Interact with local copy of all particles
        interacting_particles = particles

        # mutates in place
        dist_sampler.make_step(
                particles_to_update,
                interacting_particles,
                step_size)


        # round-robin exchange particles
        send_to_rank = (rank + 1) % num_shards
        # only send "owned" particles
        particles_to_send = particles[particle_start_idx:particle_end_idx,:]
        req = dist.isend(tensor=particles_to_send.contiguous(), dst=send_to_rank)

        # receive new particles into indices owned by other shard
        recv_from_rank = (rank - 1 + num_shards) % num_shards
        particle_start_idx, particle_end_idx = particle_idx_range(recv_from_rank)
        new_particles = torch.empty_like(particles[particle_start_idx:particle_end_idx,:])
        req2 = dist.irecv(tensor=new_particles, src=recv_from_rank)

        req.wait()
        req2.wait()

        particles[particle_start_idx:particle_end_idx,:] = new_particles

    # save results after last update
    for i in range(particle_start_idx, particle_end_idx):
        data.append(pd.Series([l+1, i, torch.tensor(particles[i]).numpy()], index=['timestep', 'particle', 'value']))
    pd.DataFrame(data).to_pickle(os.path.join(RESULTS_DIR, 'shard-{}.pkl'.format(rank)))

def init_processes(fn, backend='tcp'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend)
    fn()

if __name__ == "__main__":
    init_processes(run)
