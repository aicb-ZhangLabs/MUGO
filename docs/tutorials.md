# Tutorials

This page gives a minimal workflow to run MUGO and reproduce the main results from the paper.

## Prerequisites

- Installation completed as in [Installation](installation.md).
- Reference genome and annotations: the scripts expect **hg38** (e.g. `hg38.fa` or `hg38.ml.fa`) and a **GTF** (e.g. GENCODE v41). Set `path/to/your/folder` in the code or config to your project root so that paths like `dataset/`, `results/` resolve correctly.

## Quick Start: Single-Tissue Optimization

To run the multi-head optimization for one tissue (e.g. **Whole Blood**) with the **Borzoi** backbone and **K = 20** heads:

```bash
# From the repository root
python src/train_model/MVP_multi_head.py --tissue blood --k 20
```

Outputs (optimization logs, selected SNPs, gains) are written under `results/` in a tissue- and K-specific folder.

## Reproducing Paper Results

1. **Dataset preparation**  
   Place your gene list (e.g. `gene_3000_borzoi_gencode_v41_hg38.csv`), FASTA, and GTF under the paths used in the scripts (see `path/to/your/folder/dataset/` in the source).

2. **Run optimization**  
   Use the same commands as in the paper (e.g. `MVP_multi_head.py` for Borzoi RNA, or the corresponding script for ATAC/CAGE/other modalities). Adjust `--tissue` and `--k` as in the experiments.

3. **Baseline benchmarks**  
   Scripts under `src/baseline_benchmark/` (e.g. CADD, FunSeq2, Greedy ISM, Random Search, Saliency) can be run to reproduce the comparison tables and figures. See the respective script headers and `readme.md` in that folder for inputs and options.

For exact hyperparameters and data splits, refer to the paper and the default arguments in each script.
