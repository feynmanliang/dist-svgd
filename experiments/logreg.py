"""
Run with `python -m torch.distributed.launch --nproc_per_node=2 experiments/dist.py`
"""
import os
import shutil
import traceback

import click
import pandas as pd
from scipy.io import loadmat

import torch
import torch.distributed as dist
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from multiprocessing import Process

from definitions import DATA_DIR, RESULTS_DIR
import dsvgd
from logreg_plots import get_results_dir, make_plots

def run(rank, num_shards, dataset_name, nparticles, niter, stepsize, exchange, wasserstein):
    torch.manual_seed(rank)

    # Define model
    # Load data
    mat = loadmat(os.path.join(DATA_DIR, 'benchmarks.mat'))
    dataset = mat[dataset_name][0, 0]
    fold = 42 # use 42 train/test split

    # split #, instance, features/label
    x_train = torch.from_numpy(dataset[0][dataset[2] - 1][fold]).to(torch.float)
    t_train = dataset[1][dataset[2] - 1][fold]

    samples_per_shard = int(x_train.shape[0] / num_shards)

    d = 1 + x_train.shape[1]
    alpha_prior = Gamma(1, 1)
    w_prior = lambda alpha: MultivariateNormal(torch.zeros(x_train.shape[1]), torch.eye(x_train.shape[1]) / alpha)

    def data_idx_range(rank):
        "Returns the (start,end) indices of the range of data belonging to worker with rank `rank`"
        return (samples_per_shard * rank, samples_per_shard * (rank+1))

    def logp(shard, x):
        "Estimate of full log likelihood using partition's local data."
        # Get shard-local data
        # NOTE: this will drop data if not divisible by num_shards
        shard_start_idx, shard_end_idx = data_idx_range(shard)
        x_train_local = x_train[shard_start_idx:shard_end_idx]
        t_train_local = t_train[shard_start_idx:shard_end_idx]

        alpha = torch.exp(x[0])
        w = x[1:].reshape(-1)
        logp = alpha_prior.log_prob(alpha)
        logp += w_prior(alpha).log_prob(w)
        logp += -torch.log(1. + torch.exp(-1.*torch.mv(t_train_local * x_train_local, w))).sum()
        return logp

    def kernel(x, y):
        return torch.exp(-1.*torch.dist(x, y, p=2)**2)

    # Initialize particles
    q = Normal(0, 1)
    make_sample = lambda: q.sample((d, 1))
    particles = torch.cat([make_sample() for _ in range(nparticles)], dim=1).t()

    dist_sampler = dsvgd.DistSampler(rank, num_shards, (lambda x: logp(rank, x)), kernel, particles,
            samples_per_shard, samples_per_shard*num_shards,
            exchange_particles=exchange in ['all_particles', 'all_scores'],
            exchange_scores=exchange == 'all_scores',
            include_wasserstein=wasserstein)

    data = []
    for l in range(niter):
        if rank == 0:
            print('Iteration {}'.format(l))

        # save results right before updating particles
        for i in range(len(dist_sampler.particles)):
            data.append(pd.Series([l, torch.tensor(dist_sampler.particles[i]).numpy()], index=['timestep', 'value']))

        dist_sampler.make_step(stepsize, h=10.0)

    # save results after last update
    for i in range(len(dist_sampler.particles)):
        data.append(pd.Series([l+1, torch.tensor(dist_sampler.particles[i]).numpy()], index=['timestep', 'value']))

    pd.DataFrame(data).to_pickle(
            os.path.join(
                get_results_dir(dataset_name, num_shards, nparticles, stepsize, exchange, wasserstein),
                'shard-{}.pkl'.format(rank)))

def init_distributed(rank, dataset_name, nparticles, niter, stepsize, exchange, wasserstein):
    try:
        dist.init_process_group('tcp', rank=rank, init_method='env://')

        rank = dist.get_rank()
        num_shards = dist.get_world_size()
        run(rank, num_shards, dataset_name, nparticles, niter, stepsize, exchange, wasserstein)
    except Exception as e:
        print(traceback.format_exc())
        raise e

@click.command()
@click.option('--dataset', type=click.Choice([
    'banana', 'diabetis', 'german', 'image', 'splice', 'titanic', 'waveform']), default='banana')
@click.option('--nproc', type=click.IntRange(0,32), default=1)
@click.option('--nparticles', type=int, default=10)
@click.option('--niter', type=int, default=100)
@click.option('--stepsize', type=float, default=1e-3)
@click.option('--exchange', type=click.Choice(['partitions', 'all_particles', 'all_scores']), default='partitions')
@click.option('--wasserstein/--no-wasserstein', default=False)
@click.option('--master_addr', default='127.0.0.1', type=str)
@click.option('--master_port', default=29500, type=int)
@click.option('--plots/--no-plots', default=True)
@click.pass_context
def cli(ctx, dataset, nproc, nparticles, niter, stepsize, exchange, wasserstein, master_addr, master_port, plots):
    # clean out any previous results files
    results_dir = get_results_dir(dataset, nproc, nparticles, stepsize, exchange, wasserstein)
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    if nproc == 1:
        run(0, 1, dataset, nparticles, niter, stepsize, exchange, wasserstein)
    else:
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['WORLD_SIZE'] = str(nproc)

        processes = []
        for rank in range(nproc):
            p = Process(target=init_distributed, args=(rank, dataset, nparticles, niter, stepsize, exchange, wasserstein,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    if plots:
        ctx.forward(make_plots)


if __name__ == "__main__":
    cli()
