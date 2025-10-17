#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_late2d
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o output/output_late2d_%J.out
#BSUB -e output/output_late2d_%J.err

mkdir -p output

module load python3/3.12.11
source ~/projects/fitvid-classifier/venv/bin/activate

python ~/projects/fitvid-classifier/scripts/train_late_fusion_2d.py \
    --dataset leakage \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --patience 5