# fractal

This repository accompanies the publication

> [FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](https://arxiv.org/abs/2404.04817). *ACL (Main Conference)* (2025)

There are independent directories for each task defined in the paper:

- `preference` contains preprocessing scripts and model training scripts for
 the QA Preference Feedback experiments discussed in the paper.
- `retrieval` contains preprocessing scripts and model training scripts for
 the FirA multiclass classification and MultiSpanQA binary classification
 experiments.
- `math-reasoning` contains preprocessing scripts and model training scripts for
 the PRM800K math reasoning binary classification task.
- `entailment` contains preprocessing scripts and model training scripts for
 the WikiCatSum and AquaMuSe datasets.

## Installation

- Install [XManager](https://github.com/google-deepmind/xmanager)
- Install [Docker](https://github.com/google-deepmind/xmanager#install-docker-optional)
- (Optional, Recommended) Make a virtual environment:
```bash
cd <fractal>
python -m venv .venv
source .venv/bin/activate
```
- Run `pip install -r requirements.txt` from fractal/ folder

## Usage

Instructions for running the experiments are provided for each task separately.

- `preference`: [preference/README.md](preference/README.md)
- `retrieval`: [retrieval/MultiSpanQA/README.md](retrieval/MultiSpanQA/README.md) and [retrieval/FirA/README.md](retrieval/FirA/README.md)
- `math_reasoning`: [math_reasoning/README.md](math_reasoning/README.md)
- `entailment`: [entailment/README.md](entailment/README.md)

## Citing this work

```
@misc{makhija2024fractalfinegrainedscoringaggregate,
      title={FRACTAL: Fine-Grained Scoring from Aggregate Text Labels},
      author={Yukti Makhija and Priyanka Agrawal and Rishi Saket and Aravindan Raghuveer},
      year={2024},
      eprint={2404.04817},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.04817},
}
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
