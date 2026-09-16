# Chemical-formula-informed-compositional-relationship-learning-for-potential-mineral-prediction-

This repository contains the implementation of **MRLMRP**, a graph-based learning framework for potential mineral prediction. The model incorporates mineral compositional information and pretrained mineral representations into graph representation learning.

This implementation is developed based on **SELFRec**, an open-source framework for self-supervised recommender systems developed by Coder-Yu.

**Base framework:** `Coder-Yu/SELFRec`

## Project Structure

```text
.
├── main.py                  # Main entry point
├── SELFRec.py               # SELFRec execution framework
├── conf/
│   └── MRLMRP.conf          # Model and training configuration
├── model/
│   └── graph/
│       └── MRLMRP.py        # MRLMRP model
├── datasets/                # Training, validation, and test data
├── pretrained_emb/          # Pretrained mineral embeddings
├── base/                    # Base graph/recommendation classes
├── util/                    # Utility functions
└── results/                 # Model outputs
```

## Requirements

The main dependencies include:

* Python 3
* PyTorch
* NumPy
* FAISS
* PyTorch Geometric

Install the required packages according to your CUDA/PyTorch environment.

## Data

The default configuration uses the **Li** mineral dataset:

```text
datasets/mineral-Li/train.csv
datasets/mineral-Li/dev.csv
datasets/mineral-Li/test.csv
```

Pretrained mineral embeddings are loaded from:

```text
pretrained_emb/mineral_embeddings_64.pt
```

Model and dataset settings can be modified in:

```text
conf/MRLMRP.conf
```

## Run

Run the model with:

```bash
python main.py
```

The default experiment uses:

```text
model   = MRLMRP
dataset = Li
seed    = 42
k       = 3
coe     = 0.3
```

Experimental results are written to the `results/` directory.

## Acknowledgements

This project is built upon and adapted from:

**SELFRec: An Open-Source Framework for Self-Supervised Recommender Systems**
GitHub repository: `Coder-Yu/SELFRec`

We thank the authors of SELFRec for providing the open-source framework that serves as the foundation of this implementation.

## Citation

If you use this code in your research, please cite the corresponding paper:

> **Chemical Formula-Informed Compositional Relationship Learning for Potential Mineral Prediction**

Citation information will be updated after publication.

Please also acknowledge the original **SELFRec** framework when using code derived from it.
