# FedPVR

Official implementation of:

**FedPVR: Personalized Federated Prompt Learning under Data Heterogeneity with Visual Prototype-Guided Classifier Calibration and Reliability-Aware Inference**

This repository provides the implementation of FedPVR for personalized federated prompt learning under heterogeneous client distributions.

The implementation adopts **CLIP ViT-B/16** as the pretrained vision-language backbone. 
## Main Trainer Variants

| Trainer | Paper Variant | Description |
|---|---|---|
| `GL_SVDMSE_ADAPTIVE_PC` | Full model | Visual Prototype-Guided Classifier Calibration (VPGC) + Reliability-Aware Inference |
| `GL_SVDMSE_PC` | w/o RAI | VPGC only, without adaptive global-local fusion |
| `GL_SVDMSE_ADAPTIVE` | w/o VPGC | Adaptive global-local fusion only, without visual prototype calibration |
| `GL_SVDMSE` | FedPHA reproduced baseline | Reproduced global-local prompt learning baseline |

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The experiments are conducted with CLIP ViT-B/16 and 16-shot federated prompt learning.

## Dataset Preparation

Datasets are not included in this repository. Please download them from their official sources and place them under a dataset root directory.

The expected directory structure is:

```text
DATA/
├── caltech-101/
├── dtd/
├── food101/
├── oxford_flowers/
├── oxford_pets/
├── OFFICE31/
└── office_home/
```

The provided scripts use `--root DATA` by default.

If your datasets are stored in another location, modify the dataset root path in the scripts or specify it manually:

```bash
--root /path/to/DATA
```

## Quick Start

Run the full model on DTD:

```bash
bash scripts/run_full_dtd.sh
```

Run ablation experiments on DTD:

```bash
bash scripts/run_ablation_dtd.sh
```

Run the full model on Office31:

```bash
bash scripts/run_full_office31.sh
```

## Default Settings

- Backbone: CLIP ViT-B/16
- Shots: 16
- Communication rounds: 50
- Dirichlet beta: 0.5
- Prototype coefficient: 0.3
- Single-domain clients: 10
- Office31 clients: 6
- OfficeHome clients: 8

