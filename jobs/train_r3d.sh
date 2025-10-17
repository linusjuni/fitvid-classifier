#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_r3d
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o output_r3d_%J.out
#BSUB -e output_r3d_%J.err

module load python3/3.12.11
source ~/projects/fitvid-classifier/venv/bin/activate

python ~/projects/fitvid-classifier/scripts/train_r3d.py \
    --dataset leakage \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --patience 5