import torch
from PIL import Image
import torchvision.transforms as transforms
import sys


def main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA not available, using CPU")

    # Read image
    import os

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default: look for image in same directory as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "test_image.jpg")

    try:
        print(f"\nReading image from: {image_path}")
        image = Image.open(image_path)
        print(f"Image size: {image.size}")
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found!")
        print("Usage: python scripts/test_gpu_image.py [image_path]")
        print("Or place test_image.jpg in the scripts/ folder")
        return

    # Transform image to tensor
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to standard size
            transforms.ToTensor(),  # Convert to tensor [0, 1]
        ]
    )

    tensor = transform(image)
    print(f"\nTensor shape: {tensor.shape}")
    print(f"Tensor device (before): {tensor.device}")

    # Move tensor to GPU
    tensor_gpu = tensor.to(device)
    print(f"Tensor device (after): {tensor_gpu.device}")

    # Save tensor to disk
    output_path = "saved_tensor.pt"
    torch.save(tensor_gpu, output_path)
    print(f"\nTensor saved to: {output_path}")

    # Load it back to verify
    loaded_tensor = torch.load(output_path)
    print(f"Loaded tensor shape: {loaded_tensor.shape}")
    print(f"Loaded tensor device: {loaded_tensor.device}")

    print("\n✓ All operations completed successfully!")


if __name__ == "__main__":
    main()
