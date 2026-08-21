# FedPVR

Code implementation for the manuscript:

**FedPVR: Personalized Federated Prompt Learning under Data Heterogeneity with Visual Prototype-Guided Classifier Calibration and Reliability-Aware Inference**


The implementation uses **CLIP ViT-B/16** as the pretrained
vision-language backbone, with the pretrained encoders kept frozen during
federated prompt learning.

## Main Trainer Variants

| Trainer | Paper variant | Description |
|---|---|---|
| GL_SVDMSE_ADAPTIVE_PC | Full model | Visual Prototype-Guided Classifier Calibration (VPGC) + Adaptive Fusion |
| GL_SVDMSE_PC | w/o AF | VPGC-only, without adaptive global-local fusion |
| GL_SVDMSE_ADAPTIVE | w/o VPGC | Adaptive Fusion only, without visual prototype calibration |
| GL_SVDMSE | FedPHA reproduced baseline | Original global-local prompt learning baseline |

## Installation

`ash
pip install -r requirements.txt
`

The paper experiments use CLIP ViT-B/16 with 16-shot federated prompt learning.

## Dataset Preparation

Datasets are not included. Download them manually and place them under a dataset root, for example:

`	ext
DATA/
├── caltech-101/
├── dtd/
├── food101/
├── oxford_flowers/
├── oxford_pets/
├── OFFICE31/
└── office_home/
`

The example scripts use --root DATA by default. If your data are stored elsewhere, edit --root DATA in the scripts or run the command manually with --root /path/to/DATA.

## Quick Start

Run the full model on DTD:

`ash
bash scripts/run_full_dtd.sh
`

Run ablation variants on DTD:

`ash
bash scripts/run_ablation_dtd.sh
`

Run the full model on Office31:

`ash
bash scripts/run_full_office31.sh
`

## Default Settings

- Backbone: ViT-B/16
- Shots: 16
- Communication rounds: 50
- Dirichlet beta: 0.5
- Prototype coefficient: 0.3
- Single-domain clients: 10
- Office31 clients: 6
- OfficeHome clients: 8