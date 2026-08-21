# Running Guide

The main entry is federated_main.py.

Trainer mapping:

- GL_SVDMSE_ADAPTIVE_PC: full ProtoCalFed.
- GL_SVDMSE_PC: w/o AF, VPGC-only.
- GL_SVDMSE_ADAPTIVE: w/o VPGC, AF-only.
- GL_SVDMSE: reproduced FedPHA baseline.

Output directories are generated automatically and ignored by git.
