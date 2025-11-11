# UniPen Dataset Parser for PyTorch

This module provides a complete parser and PyTorch Dataset implementation for the UniPen handwritten character dataset.

## Features

- **Complete UniPen file parser**: Parses `.pen` files with stroke data, segments, and metadata
- **PyTorch Dataset**: Ready-to-use `Dataset` class compatible with PyTorch `DataLoader`
- **Flexible data processing**: Supports both single character and sequence labeling
- **Automatic indexing**: Utility function to build index files from dataset directory
- **Stroke normalization**: Optional coordinate normalization to [0, 1]
- **Padding/truncation**: Handles variable-length strokes with configurable max points

## Installation

Make sure you have the required dependencies:

```bash
pip install torch numpy pillow
```

## Usage

### 1. Build the Index File

First, create an index file by scanning your UniPen dataset:

```python
from unipen_class import build_index

build_index(
    root_dir="unipen/CDROM/train_r01_v07",
    output_path="unipen_index.json",
    pattern="**/*.pen"  # Glob pattern to find .pen files
)
```

### 2. Create the Dataset

```python
from unipen_class import Unipen
from torch.utils.data import DataLoader

dataset = Unipen(
    root="unipen/CDROM/train_r01_v07",
    index_path="unipen_index.json",
    alphabet="abcdefghijklmnopqrstuvwxyz",
    target_mode="char",  # or "sequence" for multi-character sequences
    max_points=512,      # Maximum number of points per sample
    normalize=True       # Normalize coordinates to [0, 1]
)
```

### 3. Use with DataLoader

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

for strokes, labels in dataloader:
    # strokes: (batch_size, max_points, 3) tensor with [x, y, pen_up]
    # labels: (batch_size,) tensor with character indices
    pass
```

## Dataset Structure

The dataset expects UniPen files (`.pen`) in a directory structure like:
```
unipen/
  CDROM/
    train_r01_v07/
      data/
        1a/
          apa/
            apa00/
              file.pen
```

## Data Format

Each sample returns:
- **strokes**: Tensor of shape `(max_points, 3)` containing `[x, y, pen_up]` for each point
- **label**: Integer index (for `target_mode="char"`) or tensor of indices (for `target_mode="sequence"`)

## API Reference

### `parse_unipen_file(file_path)`

Parses a UniPen `.pen` file and returns a dictionary with:
- `segments`: List of segment information
- `strokes`: List of stroke arrays (each with shape `(N, 3)`)
- `metadata`: Dictionary of metadata from the file
- `base_path`: Base path from `.INCLUDE` directive

### `build_index(root_dir, output_path, pattern="**/*.pen")`

Scans the dataset directory and creates a JSON index file containing all samples.

### `Unipen` Dataset Class

**Parameters:**
- `root`: Root directory of the UniPen dataset
- `index_path`: Path to JSON index file
- `alphabet`: String of characters for label encoding (default: lowercase letters)
- `target_mode`: `"char"` for single character classification, `"sequence"` for sequence labeling
- `transform`: Optional transform to apply to stroke data
- `target_transform`: Optional transform to apply to labels
- `max_points`: Maximum number of points per sample (default: 512)
- `normalize`: Whether to normalize stroke coordinates (default: True)

## Example

See `example_usage.py` for a complete working example.

