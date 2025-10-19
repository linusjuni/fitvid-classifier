#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_spatial
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o outputs/output_spatial_%J.out
#BSUB -e outputs/output_spatial_%J.err

mkdir -p outputs

module load python3/3.12.11
source ~/projects/fitvid-classifier/venv/bin/activate

python ~/projects/fitvid-classifier/scripts/train_spatial_stream.py \
    --dataset ${DATASET:-no_leakage} \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --patience 5 \
    --frame_strategy random