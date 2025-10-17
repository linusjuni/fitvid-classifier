# Convenience script to submit all training jobs at once

echo "Submitting all training jobs..."
echo ""

echo "1. Submitting AggregationModel2D..."
bsub -app c02516_1g.10gb < jobs/train_aggregation_2d.sh

echo "2. Submitting LateFusionModel2D..."
bsub -app c02516_1g.10gb < jobs/train_late_fusion_2d.sh

echo "3. Submitting EarlyFusionModel2D..."
bsub -app c02516_1g.10gb < jobs/train_early_fusion_2d.sh

echo "4. Submitting R3DModel..."
bsub -app c02516_1g.10gb < jobs/train_r3d.sh

echo ""
echo "All jobs submitted!"