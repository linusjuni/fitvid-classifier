#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_temporal
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o outputs/output_temporal_%J.out
#BSUB -e outputs/output_temporal_%J.err

mkdir -p outputs

module load python3/3.12.11
source ~/projects/fitvid-classifier/venv/bin/activate

python ~/projects/fitvid-classifier/scripts/train_temporal_stream.py \
    --dataset ${DATASET:-no_leakage} \
    --epochs 200 \
    --batch_size 32 \
    --lr 1e-5 \
    --patience 10 \