#!/bin/bash

# Test different fusion weights
for alpha in 0.3 0.4 0.5 0.6 0.7
do
    beta=$(echo "1 - $alpha" | bc -l)
    echo ""
    echo "=========================================="
    echo "Testing with alpha=$alpha (spatial=$alpha, temporal=$beta)"
    echo "=========================================="
    
    python scripts/test_two_stream_model.py \
        --spatial_checkpoint checkpoints/aggregation_2d_no_leakage/best_model.pth \
        --temporal_checkpoint checkpoints/temporal_stream_no_leakage/best_model.pth \
        --dataset no_leakage \
        --fusion_weights $alpha $beta 2>/dev/null
done