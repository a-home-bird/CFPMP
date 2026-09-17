# CFPMP: Chemical Formula-Informed Compositional Relationship Learning for Potential Mineral Prediction

This repository provides the implementation, processed datasets, and pretrained mineral representations associated with the paper:

> **Chemical Formula-Informed Compositional Relationship Learning for Potential Mineral Prediction**

CFPMP incorporates chemical formula information as a compositional prior for potential mineral prediction. The framework learns composition-aware relationships among minerals and integrates the resulting affinity information into graph-based mineral prediction models.

In the experiments reported in the paper, **MRLMRP** is used as the default graph-based backbone unless otherwise specified. Accordingly, some implementation files retain the name `MRLMRP`, while the CFPMP-specific composition-aware affinity learning components are integrated into the backbone implementation.

This implementation is developed based in part on **SELFRec**, an open-source framework for self-supervised recommender systems.

---

## 1. Repository Structure

```text
CFPMP/
├── README.md
├── LICENSE
├── main.py
├── SELFRec.py
│
├── conf/
│   └── MRLMRP.conf
│
├── model/
│   └── graph/
│       └── MRLMRP.py
│
├── datasets/
│   ├── mineral-U/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   ├── test.csv
│   │   └── mineral_list.txt
│   │
│   ├── mineral-Li/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   ├── test.csv
│   │   └── mineral_list.txt
│   │
│   ├── mineral-Ni/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   ├── test.csv
│   │   └── mineral_list.txt
│   │
│   └── mineral-Co/
│       ├── train.csv
│       ├── dev.csv
│       ├── test.csv
│       └── mineral_list.txt
│
├── pretrained_emb/
│   └── mineral_embeddings_64.pt
│
├── data/
├── base/
├── util/
└── results/
```

The main components are:

- `main.py`: main entry point for training and evaluation.
- `SELFRec.py`: execution interface adapted from SELFRec.
- `conf/MRLMRP.conf`: model, dataset, optimization, and output configuration.
- `model/graph/MRLMRP.py`: graph-based backbone together with the CFPMP downstream composition-aware affinity learning components.
- `datasets/`: processed training, validation, and test interactions for the U, Li, Ni, and Co datasets.
- `pretrained_emb/`: pretrained mineral representations derived from chemical formula information.
- `base/`: base classes used by the graph recommendation framework.
- `data/`: data-loading utilities.
- `util/`: auxiliary functions for training, evaluation, graph construction, and loss computation.
- `results/`: default directory for experimental outputs.

---

## 2. Requirements

The implementation is written in Python and uses PyTorch.

### Main Dependencies

The main software dependencies include:

```text
Python 3
PyTorch
NumPy
FAISS
PyTorch Geometric
```

Additional standard Python packages may be required by the SELFRec-based framework.

### Computational Requirements

The experiments reported in the paper were conducted using:

```text
GPU: NVIDIA GeForce RTX 3090
GPU memory: 24 GB
Deep-learning framework: PyTorch
```

A CUDA-compatible NVIDIA GPU is recommended for reproducing the experiments.

CPU execution may be possible for some components, but GPU acceleration is strongly recommended because graph training and repeated experimental runs can otherwise be substantially slower.

---

## 3. Installation

Clone the repository:

```bash
git clone https://github.com/a-home-bird/CFPMP.git
cd CFPMP
```

Create a dedicated Python environment:

```bash
conda create -n cfpmp python=3
conda activate cfpmp
```

Install PyTorch according to your CUDA environment.

Please follow the official PyTorch installation instructions to select the appropriate CUDA-compatible build.

Then install the remaining main dependencies:

```bash
pip install numpy
pip install faiss-cpu
pip install torch-geometric
```

If GPU-enabled FAISS is available and compatible with your environment, the corresponding GPU version can be used instead.

For strict reproducibility, users are encouraged to reproduce the experiments using software versions compatible with the environment documented for the final release.

---

## 4. Data

The repository contains four element-specific mineral datasets used in the experiments:

```text
datasets/mineral-U/
datasets/mineral-Li/
datasets/mineral-Ni/
datasets/mineral-Co/
```

Each dataset contains training, validation, and test interactions:

```text
train.csv
dev.csv
test.csv
mineral_list.txt
```

The datasets correspond to minerals containing the elements:

- U: uranium-bearing minerals
- Li: lithium-bearing minerals
- Ni: nickel-bearing minerals
- Co: cobalt-bearing minerals

The processed datasets are derived from mineral occurrence records described in the paper.

---

## 5. Data Format

Despite the `.csv` file extension, the interaction files are stored as whitespace-separated records.

Each interaction contains three fields:

```text
locality_id    mineral_id    weight
```

For example:

```text
0    1572    1
0    1585    1
0    924     1
0    414     1
```

where:

- `locality_id` is the integer identifier of a locality;
- `mineral_id` is the integer identifier of a mineral;
- `weight` indicates an observed locality-mineral occurrence and is set to `1` in the released datasets.

The files are loaded by the data loader in:

```text
data/loader.py
```

The `mineral_list.txt` file stores the mineral identifiers required to associate mineral nodes with the corresponding pretrained mineral representations.

---

## 6. Pretrained Chemical Formula Representations

The pretrained mineral representations used by CFPMP are provided at:

```text
pretrained_emb/mineral_embeddings_64.pt
```

The representation dimensionality is:

```text
64
```

These embeddings encode compositional information derived from chemical formulas and are used to construct composition-aware relationships among minerals in the downstream potential mineral prediction task.

The released pretrained embeddings allow the downstream CFPMP experiments to be executed without repeating the chemical formula pre-training stage.

---

## 7. Quick Start

The simplest way to run the model is:

```bash
python main.py
```

The default experiment in `main.py` uses:

```python
model = 'MRLMRP'
dataset = 'Li'
seed = 42
k = 3
coe = 0.3
```

The default configuration therefore runs the CFPMP-enhanced graph model on the **Li dataset**.

During execution, the program:

1. loads the locality-mineral training, validation, and test interactions;
2. loads the pretrained mineral representations;
3. constructs the locality-mineral graph;
4. constructs and refines the composition-aware mineral affinity information;
5. integrates the affinity information into the graph-based backbone;
6. trains the model;
7. evaluates the model on held-out interactions; and
8. reports prediction performance and running time.

---

## 8. Running Experiments on Different Datasets

The dataset can be changed through the `dataset` variable in `main.py`.

For example:

```python
dataset = 'U'
```

for the U dataset,

```python
dataset = 'Li'
```

for the Li dataset,

```python
dataset = 'Ni'
```

for the Ni dataset, or

```python
dataset = 'Co'
```

for the Co dataset.

The program automatically replaces the corresponding dataset paths defined in the configuration.

For example, when:

```python
dataset = 'Li'
```

the following files are used:

```text
datasets/mineral-Li/train.csv
datasets/mineral-Li/dev.csv
datasets/mineral-Li/test.csv
datasets/mineral-Li/mineral_list.txt
```

---

## 9. Configuration

The primary configuration file is:

```text
conf/MRLMRP.conf
```

The current configuration contains the following main options:

```text
training.set
test.set
dev.set
model.name
model.type
item.ranking
embedding.size
num.max.epoch
batch_size
learnRate
reg.lambda
output.setup
seed
mineral_embedding_path
mineral_dict_path
```

The default values include:

```text
embedding.size = 64
num.max.epoch = 300
batch_size = 2048
learnRate = 0.001
reg.lambda = 0.0001
seed = 42
```

The pretrained mineral representations are loaded from:

```text
pretrained_emb/mineral_embeddings_64.pt
```

and the default output directory is:

```text
results/
```

---

## 10. CFPMP-Related Parameters

Two parameters used by the released downstream implementation are defined in `main.py`:

```python
k = 3
coe = 0.3
```

### `k`

`k` controls the number of neighboring minerals retained when constructing the sparsified mineral affinity graph.

The paper evaluates:

```text
k ∈ {1, 3, 5, 7, 9}
```

and the default setting is:

```text
k = 3
```

### `coe`

`coe` controls the relative contribution of the two affinity components used in the released implementation.

The default value is:

```text
coe = 0.3
```

This parameter can be changed in `main.py` when conducting sensitivity experiments.

---

## 11. Model Inputs

The main inputs to the downstream CFPMP implementation are:

### Locality-Mineral Interactions

```text
datasets/mineral-*/train.csv
datasets/mineral-*/dev.csv
datasets/mineral-*/test.csv
```

These files define the observed locality-mineral graph and the held-out interactions used for validation and testing.

### Mineral Identifiers

```text
datasets/mineral-*/mineral_list.txt
```

This file is used to associate mineral nodes with pretrained representations.

### Pretrained Mineral Representations

```text
pretrained_emb/mineral_embeddings_64.pt
```

These representations provide the compositional prior used by CFPMP.

---

## 12. Model Outputs

During training, the program prints training progress and evaluation information to the terminal.

After model training, the implementation reports prediction metrics and total running time.

The default configuration stores experimental outputs under:

```text
results/
```

The exact output format is determined by the SELFRec-based evaluation framework and the settings in:

```text
conf/MRLMRP.conf
```

---

## 13. Expected Behaviour

A typical CFPMP experiment follows the workflow:

```text
processed locality-mineral data
            +
pretrained mineral representations
            ↓
load locality-mineral graph
            ↓
construct composition-aware mineral affinity structure
            ↓
task-adaptive affinity refinement
            ↓
graph-based representation learning
            ↓
potential mineral prediction
            ↓
evaluation on held-out locality-mineral interactions
```

Under the default configuration, running:

```bash
python main.py
```

should train and evaluate the model on the Li dataset and print the resulting evaluation metrics and running time.

Exact numerical values can vary slightly across different:

- GPUs;
- CUDA versions;
- PyTorch versions;
- PyTorch Geometric versions;
- FAISS implementations; and
- random-number generation environments.

For comparisons with the paper, the same dataset split, hyperparameters, random seeds, and software environment should be used.

---

## 14. Reproducing the Main Results

The main potential mineral prediction experiments can be reproduced using the processed datasets and pretrained mineral representations included in this repository.

For each of the four datasets:

```text
U
Li
Ni
Co
```

perform the following procedure.

Select the dataset in `main.py`, for example:

```python
dataset = 'Li'
```

Set the desired random seed:

```python
seed = 42
```

Set the affinity-related parameters:

```python
k = 3
coe = 0.3
```

Then run:

```bash
python main.py
```

Repeat the experiment using the required random seeds when reproducing the multi-run results reported in the paper.

The paper reports experimental performance averaged over multiple independent runs.

---

## 15. Hyperparameter Experiments

The top-neighbor parameter can be changed in `main.py`:

```python
k = 1
k = 3
k = 5
k = 7
k = 9
```

The affinity-fusion coefficient can likewise be changed:

```python
coe = 0.1
coe = 0.3
coe = 0.5
coe = 0.7
coe = 0.9
```

These settings can be used to reproduce the corresponding parameter-sensitivity experiments.

---

## 16. Reproducibility

This repository provides the main materials required for reproducing the downstream potential mineral prediction experiments, including:

- processed U, Li, Ni, and Co datasets;
- fixed training, validation, and test splits;
- pretrained mineral representations;
- graph-based model implementation;
- CFPMP downstream affinity-learning components;
- model configuration;
- training and evaluation code; and
- hyperparameter settings.

The chemical formula representations required by the downstream experiments are provided directly in:

```text
pretrained_emb/mineral_embeddings_64.pt
```

Therefore, users can reproduce the downstream CFPMP experiments without repeating the chemical formula pre-training stage.

If numerical results differ from those reported in the paper, please first verify:

```text
dataset
random seed
embedding dimensionality
k
coe
CUDA version
PyTorch version
PyTorch Geometric version
FAISS version
GPU hardware
```

---



## 18. Extending the Model

The CFPMP strategy is designed to provide composition-aware mineral information to graph-based potential mineral prediction models.

Researchers interested in extending the implementation can modify:

```text
model/graph/
```

to integrate composition-aware affinity information into other graph-based backbones.

The released implementation uses MRLMRP as the default backbone because this is the main configuration adopted in the paper.

---

## 19. Code Documentation

The public repository is intended to provide source-code documentation and comments in English to facilitate reproducibility and reuse.

Before using or redistributing modified versions of the implementation, contributors are encouraged to maintain English-language comments and documentation throughout the codebase.

---

## 20. Data and Code Availability

The source code, processed datasets, configuration files, and pretrained mineral representations associated with the study are publicly available in this repository:

```text
https://github.com/a-home-bird/CFPMP
```

The manuscript also provides a direct link to this repository in the Code and Data Availability section.

All materials are provided as normal repository files and directories rather than as a single compressed archive.

---

## 21. Acknowledgements

This implementation is developed in part based on:

**SELFRec: An Open-Source Framework for Self-Supervised Recommender Systems**

Original repository:

```text
https://github.com/Coder-Yu/SELFRec
```

We thank the authors of SELFRec for releasing the open-source framework that provides part of the infrastructure used in this implementation.

Users of this repository should also acknowledge the original SELFRec project where appropriate.

---

## 22. License

The source code in this repository is distributed under the license specified in:

```text
LICENSE
```

Please consult the `LICENSE` file for permissions, conditions, and limitations.

Because part of the implementation is adapted from SELFRec, the selected license for this repository should also comply with the licensing terms of the original SELFRec project.

---

## 23. Citation

If you use CFPMP, the processed datasets, or the released mineral representations in your research, please cite:

```text
Chemical Formula-Informed Compositional Relationship Learning
for Potential Mineral Prediction
```

Full bibliographic information will be added after publication.

A BibTeX entry will be provided after the paper is formally published.

---

## 24. Contact

For questions regarding the implementation or experiments, please contact the authors through the contact information provided in the corresponding paper.
