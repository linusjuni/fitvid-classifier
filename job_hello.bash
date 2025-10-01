#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J hello_world
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 10:00
#BSUB -o hello_%J.out
#BSUB -e hello_%J.err

module load python3/3.12.11
source ~/projects/fitvid-classifier/venv/bin/activate

python ~/projects/fitvid-classifier/scripts/hello.py