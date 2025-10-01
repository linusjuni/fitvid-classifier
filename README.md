# Virtual Environment Setup

## First Time Setup

Assuming you have a `projects` directory.

```bash
cd ~/projects/fitvid-classifier
module load python3/3.12.11
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Daily Usage

Every time you log into the HPC:

```bash
cd ~/projects/fitvid-classifier
module load python3/3.12.11
source venv/bin/activate
```

## Deactivate

When you're done working:

```bash
deactivate
```

## Verify Installation

```bash
python -c 'import torch, numpy, pandas, polars, sklearn, matplotlib, seaborn; print("All packages work!")'
```
