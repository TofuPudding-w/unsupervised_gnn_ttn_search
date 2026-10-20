# Multivariate quantum state preparation with optimized tensor networks

This repository contains research code for the paper 'Multivariate quantum state preparation with optimized tensor networks' ([arXiv:2609.09304](https://arxiv.org/abs/2609.09304)). It contains functionality for:
- Quantics function representations
- Quantics index settings and mappings
- Tensor operations:
    * Smart contraction and index tracking
    * Matricization
    * Factorizations: SVD and interpolative decomposition
    * Traces and partial traces
- Tree tensor networks
    * Internal graph structure
    * Gauge transformations (orthogonality centres, interpolative gauge)
    * Gauge metadata (i.e. index settings in interpolative gauge)
    * Common ansätze (MPS, comb network)
- Tensor cross-interpolation generalized to TTNs, implementing the approach of [arXiv:2410.03572](https://arxiv.org/abs/2410.03572).
- Spectral Clustering for Entanglement Minimizing Trees (SCENT), as introduced in our paper.
    * Computing affinity matrices (Monte Carlo and tensor network contraction)
    * Spectral clustering to determine tree structure agglomeratively

## Installation

This library has two specific dependencies: `scipy` of version below 1.15.0 (as this version partially broke the interpolative decomposition), and `graphviz`. I recommend using the included `environment.yml` to set up a custom conda environment, as it's set up with all the right dependencies.

After cloning the repository, use `cd` to enter the main directory (e.g. `cd /home/username/treetensorcomp`). Set up a compatible conda environment by running
```
conda env create -f environment.yml
```
I recommend configuring the library in development mode so you can make changes as you work:
```
conda-develop /home/username/treetensorcomp
```
replacing the directory with wherever you cloned it. Finally, if you don't already have `graphviz` installed, you'll need to install it ([instructions here](https://graphviz.org/download/), platform-dependent) in order to use the `.draw()` feature for `TTNGraph` instances.

## Getting started

Some examples are given in `demo.ipynb` to get you started. If anything breaks or doesn't make sense, please feel free to contact me ([matt.sims-goh@quantummotion.tech](mailto:matt.sims-goh@quantummotion.tech)).
