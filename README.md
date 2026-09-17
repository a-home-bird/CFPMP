# CFPMP: Chemical Formula-Informed Compositional Relationship Learning for Potential Mineral Prediction

This repository provides the implementation, processed datasets, and pretrained mineral representations associated with the paper:

> **Chemical Formula-Informed Compositional Relationship Learning for Potential Mineral Prediction**

CFPMP incorporates chemical formula information as a compositional prior for potential mineral prediction. In the experiments reported in the paper, **MRLMRP** is used as the default graph-based backbone. Some implementation files therefore retain the name `MRLMRP`, while the CFPMP-specific composition-aware affinity learning components are integrated into the backbone implementation.

This project is developed in part based on **SELFRec**, an open-source framework for self-supervised recommender systems.

---

## Repository Structure

```text
CFPMP/
├── README.md
├── main.py
├── SELFRec.py
├── conf/
│   └── MRLMRP.conf
├── model/
│   └── graph/
│       └── MRLMRP.py
├── datasets/
│   ├── mineral-U/
│   ├── mineral-Li/
│   ├── mineral-Ni/
│   └── mineral-Co/
├── pretrained_emb/
│   └── mineral_embeddings_64.pt
├── data/
├── base/
├── util/
└── results/
```

Main files and directories:

- `main.py`: training and evaluation entry point.
- `conf/MRLMRP.conf`: model and training configuration.
- `model/graph/MRLMRP.py`: MRLMRP backbone with CFPMP-related affinity learning components.
- `datasets/`: processed U, Li, Ni, and Co datasets.
- `pretrained_emb/`: pretrained mineral representations.
- `results/`: default output directory.

---

## Requirements

Main dependencies:

```text
Python 3
PyTorch
NumPy
FAISS
PyTorch Geometric
```

The experiments reported in the paper were conducted using a single **NVIDIA GeForce RTX 3090 GPU with 24 GB memory** under PyTorch. A CUDA-compatible NVIDIA GPU is recommended for reproducing the experiments.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/a-home-bird/CFPMP.git
cd CFPMP
```

Create a Python environment:

```bash
conda create -n cfpmp python=3
conda activate cfpmp
```

Install PyTorch according to your CUDA environment, then install the remaining dependencies:

```bash
pip install numpy
pip install faiss-cpu
pip install torch-geometric
```

---

## Data

The repository contains four element-specific datasets:

```text
datasets/mineral-U/
datasets/mineral-Li/
datasets/mineral-Ni/
datasets/mineral-Co/
```

Each dataset contains:

```text
train.csv
dev.csv
test.csv
mineral_list.txt
```

The interaction files are whitespace-separated and contain three fields:

```text
locality_id    mineral_id    weight
```

Example:

```text
0    1572    1
0    1585    1
0    924     1
```

where `locality_id` denotes a locality, `mineral_id` denotes a mineral, and `weight=1` indicates an observed locality-mineral occurrence.

---

## Pretrained Mineral Representations

The pretrained mineral representations used by CFPMP are provided at:

```text
pretrained_emb/mineral_embeddings_64.pt
```

The embedding dimensionality is 64. These representations encode chemical-formula-derived compositional information and are used to construct composition-aware mineral relationships.

The released embeddings allow the downstream CFPMP experiments to be reproduced without rerunning the chemical formula pre-training stage.

---

## Quick Start

Run the default experiment with:

```bash
python main.py
```

The default settings in `main.py` are:

```python
model = 'MRLMRP'
dataset = 'Li'
seed = 42
k = 3
coe = 0.3
```

The default configuration runs CFPMP with the MRLMRP backbone on the Li dataset.

To use another dataset, change:

```python
dataset = 'U'   # or 'Li', 'Ni', 'Co'
```

---

## Configuration

The main configuration file is:

```text
conf/MRLMRP.conf
```

Important options include:

```text
training.set
dev.set
test.set
embedding.size
num.max.epoch
batch_size
learnRate
reg.lambda
mineral_embedding_path
mineral_dict_path
output.setup
```

Default settings include:

```text
embedding.size = 64
num.max.epoch = 300
batch_size = 2048
learnRate = 0.001
reg.lambda = 0.0001
seed = 42
```

Two CFPMP-related parameters are defined in `main.py`:

```python
k = 3
coe = 0.3
```

`k` controls the number of retained mineral neighbors in the sparsified affinity graph, while `coe` controls the relative contribution of the affinity components.

---

## Inputs and Outputs

Main inputs:

```text
datasets/mineral-*/train.csv
datasets/mineral-*/dev.csv
datasets/mineral-*/test.csv
datasets/mineral-*/mineral_list.txt
pretrained_emb/mineral_embeddings_64.pt
```

The model uses observed locality-mineral interactions together with pretrained mineral representations to learn composition-aware mineral relationships and rank candidate minerals for each locality.

Training progress and evaluation results are printed to the terminal. Experimental outputs are stored in:

```text
results/
```

---

## Reproducing the Main Results

To reproduce the main downstream CFPMP experiments:

1. Select one of the four datasets in `main.py`:
   ```python
   dataset = 'Li'
   ```
2. Set the random seed and affinity parameters:
   ```python
   seed = 42
   k = 3
   coe = 0.3
   ```
3. Run:
   ```bash
   python main.py
   ```
4. Repeat the experiment using the required random seeds to reproduce the multi-run results reported in the paper.

For the hyperparameter study, use:

```text
k ∈ {1, 3, 5, 7, 9}
coe ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
```

Exact numerical results may vary slightly across hardware, CUDA versions, PyTorch versions, PyTorch Geometric versions, and FAISS implementations.

---

## Reproducibility Notes

This repository provides:

- processed U, Li, Ni, and Co datasets;
- fixed training, validation, and test splits;
- pretrained mineral representations;
- model implementation;
- configuration files; and
- training and evaluation code.

The downstream CFPMP experiments can be reproduced directly using the released pretrained mineral representations.

All public code comments and documentation should be kept in English.

---

## Data and Code Availability

The source code, processed datasets, configuration files, and pretrained mineral representations are available at:

```text
https://github.com/a-home-bird/CFPMP
```

All materials are provided as normal repository files and directories rather than as a single compressed archive.

---

## Acknowledgements

This implementation is developed in part based on:

**SELFRec: An Open-Source Framework for Self-Supervised Recommender Systems**

Original repository:

```text
https://github.com/Coder-Yu/SELFRec
```

We thank the authors of SELFRec for providing the open-source framework used as part of this implementation.

---

## Citation

If you use CFPMP, the processed datasets, or the released mineral representations in your research, please cite:

```text
Chemical Formula-Informed Compositional Relationship Learning
for Potential Mineral Prediction
```

Full bibliographic information will be added after publication.

---

## Contact

For questions regarding the implementation or experiments, please contact the authors through the contact information provided in the corresponding paper.
