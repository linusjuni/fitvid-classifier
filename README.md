# HPC Setup and Usage Guide

## Project Structure
`src/fitvid_classifier/` contains the reusable library code (models, utilities, etc). `scripts/` contains executable entry points that import from the library and run training/evaluation on HPC.

## First Time Setup

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

When done:

```bash
deactivate
```

## Verify Installation

```bash
python -c 'import torch, torchvision, numpy, pandas, polars, sklearn, matplotlib, seaborn; print("All packages work!")'
```

## Running Scripts

### Interactive (for testing/debugging)

**Never run code on the login node!** Request an interactive GPU node:

```bash
02516sh
```

Once on the GPU node:

```bash
module load python3/3.12.11
source venv/bin/activate
python scripts/your_script.py
```

Check GPU status: `nvidia-smi`

**Limitations:**
- Only 1 interactive session at a time
- Check jobs: `bstat`
- Kill job: `bkill JOBID`
- If you close terminal, job dies
- Exit when done: `exit`

### Batch Jobs (for long-running tasks)

Create a job script `job_script.sh`:

```bash
#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J my_job_name
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o output_%J.out
#BSUB -e output_%J.err

module load python3/3.12.11
source ~/projects/fitvid-classifier/venv/bin/activate

python ~/projects/fitvid-classifier/scripts/your_script.py
```

Submit the job:

```bash
bsub -app c02516_1g.10gb < job_script.sh
```

Monitor and check results:

```bash
bstat              # Check status
bjobs              # List jobs
bkill JOBID        # Kill a job

# After completion:
cat output_*.out   # View output
cat output_*.err   # View errors
```

**Key parameters:**
- `-W 12:00` = 12 hours max
- `-R "rusage[mem=20GB]"` = 20GB RAM
- `-n 4` = 4 CPU cores
- `%J` in filenames = replaced with job ID

**Advantages over interactive:**
- Runs in background
- Can disconnect and job continues
- Better for long training runs
- Automatic output logging