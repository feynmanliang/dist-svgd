# Experiments in distributed Stein's Variational Gradient Descent

This repo contains code which performs some experiments on SVGD
(see https://arxiv.org/abs/1608.04471).

## Reproducing experiments

Get set up, installing local directy as a python module
```bash
pip install -r requirements.txt
pip install -e .
```

Run the single machine experiments
```bash
python experiments/logreg.py
```

To run a distributed experiment
```bash
python -m torch.distributed.launch --nproc_per_node=2 experiments/dist.py
```
This will output results from each worker in `experiments/results/`.
To generate plots following this
```bash
python experiments/dist-plots.py
```

## Project Organization
 - `dsvgd/` : code for the sampler
 - `experiments/` : experiments which use the sampler
 - `data/` : datasets used in experiments
 - `figures/` : figures produced by experiments

## References
 - `benchmarks.mat` downloaded from Dr Gavin Cawley's web page (http://theoval.cmp.uea.ac.uk/matlab/default.html), distributed under GPL v3.
