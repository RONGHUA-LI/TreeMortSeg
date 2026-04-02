# TreeMortSeg


## Overview

This repository provides the code for paper "**TreeMortSeg: A Multi-Task Learning Network for Individual Tree Mortality Segmentation from High-Resolution Aerial Imagery**".

We propose TreeMortSeg, a multi-task learning framework for tree mortality segmentation. The framework incorporates structure-aware and edge-aware auxiliary tasks, each equipped with dedicated decoders to extract morphological and boundary features. These features are further integrated as spatial priors to guide the primary segmentation task, thereby improving segmentation performance for dead trees.


## Repository Structure

```text
├── main.py                   # Entrypoint: parse args, load config, run train & eval 
├── predict.py
├── eval_only.py
├── configs/                  # YAML experiment configs  
│   ├── debug.yaml
│   └── res_random.yaml
├── data_loader/              # Tile loading & split implementations  
│   ├── __init__.py
│   ├── utils.py
│   ├── random_split.py
├── models/                   # Model factory & builders  
│   ├── __init__.py
│   ├── treemortseg.py
│   ├── unet.py
│   ├── deeplab.py
│   ├── efficientunet.py
│   ├── segformer.py
│   ├── mask2former.py
├── exps/                     # Training & evaluation loops  
│   ├── train.py
│   └── evaluate.py
└── utils/                    # Misc helpers (logging, config I/O, seed control)  
    └── tools.py
```

## Citation

If you use this repository , please cite:

>  
