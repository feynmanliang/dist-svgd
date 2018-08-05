# Experiments in distributed Stein's Variational Gradient Descent

This repo contains code which performs some experiments on SVGD
(see https://arxiv.org/abs/1608.04471).

## Reproducing experiments

Get set up
```bash
pip install -r requirements.txt
```

Run the experiments
```bash
python experiments/logreg.py
```

## Project Organization
 - `dsvgd/` : code for the sampler
 - `experiments/` : experiments which use the sampler
 - `data/` : datasets used in experiments
 - `figures/` : figures produced by experiments

## References
 - `benchmarks.mat` downloaded from Dr Gavin Cawley's web page (http://theoval.cmp.uea.ac.uk/matlab/default.html), distributed under GPL v3.
