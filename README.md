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

## Running Scripts with GPU

**Never run code on the login node!** Request an interactive GPU node first:
```bash
02516sh
```

Once on the GPU node, activate your environment and run your script:

```
module load python3/3.12.11
source venv/bin/activate
python path/to/your/script.py
```

Important:

- Only 1 interactive session at a time
- Check your jobs: `bstat`
- Kill a job if needed: `bkill JOBID`
- Exit the GPU node when done: `exit`
- To check the GPU status: `nvidia-smi`