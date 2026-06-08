# MUGO: Multi-Head Genomic Optimization

[![PyPI version](https://img.shields.io/pypi/v/mugo.svg)](https://pypi.org/project/mugo/)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://mugo-framework.netlify.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MUGO** is a differentiable combinatorial optimization framework designed for discovering causal variants in the non-coding genome. By leveraging **Gumbel-Softmax relaxation** and **Straight-Through Estimators (STE)**, MUGO enables end-to-end gradient-based optimization on discrete DNA sequences.

## Key Features

* 🧬 **Model-Agnostic**: Compatible with Borzoi, Enformer, HyenaDNA, and other PyTorch-based genomic models.
* 🎯 **Multi-Modal Objectives**: Optimize for Gene Expression, Chromatin Accessibility (ATAC), or TF Binding.
* 📉 **Variance Reduction**: Built-in Multi-Head Consensus strategy to filter stochastic noise.
* 🚀 **Production Ready**: Easy-to-use Python API for high-performance computing.

## Installation

```bash
pip install mugo
```

## Quick Start
```python
import torch
from mugo import MultiHeadSelector

# Initialize the optimizer
selector = MultiHeadSelector(num_snps=1000, snp_positions=positions, k=20)

# Optimization loop
for step in range(200):
    input_seq, mask, _ = selector(ref_seq, alt_seq, tau=1.0)
    # ... compute loss and backward ...
```

## Documentation
Comprehensive documentation, tutorials, and API references are available at: 👉 https://mugo-framework.netlify.app

## Citations
If you use MUGO in your research, please cite:
@misc{mugo2026,
  author = {{SciML Team}},
  title = {MUGO: Differentiable Combinatorial Optimization for Genomics},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/anonymous/mugo}},
  note = {Accessed: 2026-01-10}
}