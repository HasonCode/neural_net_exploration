"""
Example usage of the UniPen dataset parser and PyTorch Dataset.

This script demonstrates how to:
1. Build an index file from the UniPen dataset
2. Create a PyTorch Dataset
3. Use DataLoader to batch the data
"""

from unipen_class import Unipen, build_index
from torch.utils.data import DataLoader

# Step 1: Build the index file (only need to do this once)
# This scans the dataset directory and creates a JSON index
if __name__ == "__main__":
    # Paths
    root_dir = "unipen_data/unipen/CDROM/train_r01_v07/data"
    index_path = "unipen_index.json"
    
    # Build index (comment out after first run)
    print("Building index file...")
    build_index(root_dir, index_path, pattern="**/*.dat")
    
    # Step 2: Create the dataset
    print("\nCreating dataset...")
    dataset = Unipen(
        root=root_dir,
        index_path=index_path,
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()-_=+[]{}|,.<>?;:`\\",
        target_mode="char",  # or "sequence" for multi-character sequences
        max_points=512,  # Maximum number of points per sample
        normalize=True  # Normalize coordinates to [0, 1]
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Step 3: Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Set to > 0 for multi-process loading
    )
    
    # Step 4: Iterate through batches
    print("\nLoading a batch...")
    for batch_idx, (strokes, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Strokes shape: {strokes.shape}")  # (batch_size, max_points, 3)
        print(f"  Labels shape: {labels.shape}")   # (batch_size,) for char mode
        print(f"  Labels: {labels}")
        
        # Example: Get a single sample
        if batch_idx == 0:
            sample_strokes, sample_label = dataset[0]
            print(f"\nSingle sample:")
            print(f"  Strokes shape: {sample_strokes.shape}")
            print(f"  Label: {sample_label}")
            print(f"  Label character: {dataset.alphabet[sample_label]}")
        
        # Only show first batch for demo
        if batch_idx >= 0:
            break

