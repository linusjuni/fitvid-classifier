import torch
from src.models import AggregationModel2D, LateFusionModel2D, EarlyFusionModel2D, R3DModel


def test_model(model_name, model, input_shape, expected_output_shape):
    """Test a single model with random input."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    # Create random input
    x = torch.randn(input_shape)
    print(f"Input shape: {list(x.shape)}")
    
    # Forward pass
    try:
        model.eval()
        with torch.no_grad():
            output = model(x)
        print(f"Output shape: {list(output.shape)}")
        print(f"Expected shape: {expected_output_shape}")
        
        # Check output shape
        assert list(output.shape) == expected_output_shape, \
            f"Shape mismatch! Got {list(output.shape)}, expected {expected_output_shape}"
        
        print(f"✅ {model_name} works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} failed with error:")
        print(f"   {str(e)}")
        return False


def main():
    print("\n" + "="*60)
    print("TESTING ALL VIDEO CLASSIFICATION MODELS")
    print("="*60)
    
    batch_size = 2
    num_classes = 10
    num_frames = 10
    height, width = 224, 224
    
    results = []
    
    # Test 1: AggregationModel2D (per-frame)
    model = AggregationModel2D(num_classes=num_classes, pretrained=False)
    success = test_model(
        model_name="AggregationModel2D (per-frame)",
        model=model,
        input_shape=(batch_size, 3, height, width),  # Single frames
        expected_output_shape=[batch_size, num_classes]
    )
    results.append(("AggregationModel2D", success))
    
    # Test 2: LateFusionModel2D
    model = LateFusionModel2D(num_classes=num_classes, pretrained=False)
    success = test_model(
        model_name="LateFusionModel2D",
        model=model,
        input_shape=(batch_size, 3, num_frames, height, width),  # [B, C, T, H, W]
        expected_output_shape=[batch_size, num_classes]
    )
    results.append(("LateFusionModel2D", success))
    
    # Test 3: EarlyFusionModel2D
    model = EarlyFusionModel2D(num_classes=num_classes, num_frames=num_frames, pretrained=False)
    success = test_model(
        model_name="EarlyFusionModel2D",
        model=model,
        input_shape=(batch_size, 3, num_frames, height, width),  # [B, C, T, H, W]
        expected_output_shape=[batch_size, num_classes]
    )
    results.append(("EarlyFusionModel2D", success))
    
    # Test 4: R3DModel (use smaller resolution for 3D CNN)
    model = R3DModel(num_classes=num_classes, pretrained=False)
    success = test_model(
        model_name="R3DModel (3D CNN)",
        model=model,
        input_shape=(batch_size, 3, num_frames, 112, 112),  # [B, C, T, H, W] - smaller res
        expected_output_shape=[batch_size, num_classes]
    )
    results.append(("R3DModel", success))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for model_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {model_name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All models passed! Ready to train.")
    else:
        print("\n⚠️  Some models failed. Check errors above.")
    
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)