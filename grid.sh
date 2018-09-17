#!/usr/bin/env bash
for dataset in banana diabetis german image splice titanic waveform; do
  for fold in $(seq 1 100); do
    for nproc in 1 2 4 8; do
      for exchange in partitions all_particles all_scores; do
        time ipython experiments/logreg.py -- --dataset=$dataset --fold=$fold --nproc=$nproc --nparticles=50 --niter=500 \
          --exchange=$exchange --no-wasserstein --plots
        time ipython experiments/logreg.py -- --dataset=$dataset --fold=$fold --nproc=$nproc --nparticles=50 --niter=500 \
          --exchange=$exchange --wasserstein --plots
      done
    done
  done
done
