import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
import re
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class UnipenSample:
    file_path: str
    segment_id: int
    label: str
    writer_id: Optional[str] = None
    subset: Optional[str] = None


def parse_unipen_file(file_path: Union[str, Path]) -> Dict:
    """
    Parse a UniPen .pen file and extract stroke data and segments.
    
    Args:
        file_path: Path to the .pen file
        
    Returns:
        Dictionary containing:
            - segments: List of segment information
            - strokes: List of stroke data (x, y, pen_up)
            - metadata: Dictionary of metadata from the file
    """
    file_path = Path(file_path)
    segments = []
    strokes = []
    current_stroke = []
    relevant_strokes = []
    metadata = {}
    base_path = None
    
    with open(str(file_path), 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "data" in str(file_path):
                if line.startswith('.'):
                    # Process commands
                    if line.startswith('.INCLUDE'):
                        parts = line.split()
                        if len(parts) > 1:
                            base_path = parts[1].strip()
                    elif line.startswith('.SEGMENT'):
                        parts = line.split()
                        if len(parts) >= 4:
                            segments.append({
                                'type': parts[0],
                                'number': int(parts[1]) if parts[1].isdigit() else None,
                                'question_mark': parts[2],
                                'real_text': parts[3] if len(parts) > 3 else None,
                                'file_name':file_path,
                            })
                            relevant_strokes.append(int(parts[1]) if parts[1].isdigit() else None )
    if base_path:
        with open(str(Path("unipen_data/unipen/CDROM/train_r01_v07/include")/base_path),"r", encoding="utf-8") as f:
            stroke_count = 0
            pen_state = -1
            stroke_data = []
            for line in f:
                line = line.strip()
                if line.startswith(".PEN_UP"):
                    stroke_count+=1
                    pen_state=1
                    if len(stroke_data)>0:
                        strokes.append(stroke_data)
                elif line.startswith(".PEN_DOWN"):
                    pen_state=0
                elif stroke_count in relevant_strokes and not line.startswith(".") and pen_state>=0:
                    stroke_data.append([int(line.split()[0]),int(line.split()[1]), pen_state, stroke_count])  
            # Add final stroke if exists
    return {
        'segments': segments,
        'strokes': strokes,
        'metadata': metadata,
        'base_path': base_path
    }


def strokes_to_tensor(strokes: List[np.ndarray], max_points: int = 512, normalize: bool = True) -> torch.Tensor:
    """
    Convert list of strokes to a PyTorch tensor.
    
    Args:
        strokes: List of stroke arrays, each with shape (N, 3) where columns are [x, y, pen_up]
        max_points: Maximum number of points to include (padding/truncation)
        normalize: Whether to normalize coordinates to [0, 1]
        
    Returns:
        Tensor of shape (max_points, 3) with [x, y, pen_up] for each point
    """
    # Concatenate all strokes into a single sequence
    if not strokes:
        return torch.zeros((max_points, 3), dtype=torch.float32)
    
    # Combine all strokes
    all_points = np.concatenate(strokes, axis=0)
    
    if normalize and len(all_points) > 0:
        # Normalize coordinates to [0, 1]
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        
        if x_max > x_min:
            all_points[:, 0] = (all_points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            all_points[:, 1] = (all_points[:, 1] - y_min) / (y_max - y_min)
    
    # Pad or truncate to max_points
    if len(all_points) > max_points:
        all_points = all_points[:max_points]
    else:
        padding = np.zeros((max_points - len(all_points), 3), dtype=np.float32)
        all_points = np.concatenate([all_points, padding], axis=0)
    
    return torch.from_numpy(all_points).float()


def build_index(root_dir: Union[str, Path], output_path: Union[str, Path], 
                pattern: str = "**/*.dat") -> None:
    """
    Build an index file by scanning for .pen files in the dataset.
    
    Args:
        root_dir: Root directory of the UniPen dataset
        output_path: Path to save the JSON index file
        pattern: Glob pattern to find .pen files
    """
    root_dir = Path(root_dir)
    output_path = Path(output_path)
    
    samples = []
    
    # Find all .pen files
    dat_files = list(root_dir.glob(pattern))
    print(root_dir)
    print(f"Number of dat files {len(dat_files)}")

    for pen_file in dat_files:
        try:
            # Parse the file to get segments
            parsed = parse_unipen_file(pen_file)
            
            # Extract relative path
            rel_path = pen_file.relative_to(root_dir)
            
            # Extract writer_id and subset from path
            parts = rel_path.parts
            writer_id = None
            subset = None
            
            # Typical structure: data/1a/apa/apa00/file.pen
            if len(parts) >= 3:
                subset = parts[-3] if parts[-3] in ['1a', '1b', '1c'] else None
            if len(parts) >= 2:
                writer_id = parts[-2]
            
            # Create a sample for each segment
            for seg_idx, segment in enumerate(parsed['segments']):
                # Extract label from segment string_number or metadata
                label = segment.get('string_number', '')
                if not label and 'label' in parsed['metadata']:
                    label = parsed['metadata']['label']
                
                samples.append({
                    'file_path': str(rel_path),
                    'segment_id': seg_idx,
                    'label': label,
                    'writer_id': writer_id,
                    'subset': subset
                })
        except Exception as e:
            print(f"Warning: Could not parse {pen_file}: {e}")
            continue
    
    # Save index
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Created index with {len(samples)} samples at {output_path}")


class Unipen(Dataset):
    """
    PyTorch Dataset for UniPen handwritten character dataset.
    
    Args:
        root: Root directory of the UniPen dataset
        index_path: Path to JSON index file (use build_index to create)
        alphabet: String of characters for label encoding (default: lowercase letters)
        target_mode: "char" for single character classification, "sequence" for sequence labeling
        transform: Optional transform to apply to stroke data
        target_transform: Optional transform to apply to labels
        max_points: Maximum number of points per sample (for padding/truncation)
        normalize: Whether to normalize stroke coordinates
    """
    def __init__(self, root: Union[str, Path], index_path: Union[str, Path], 
                 alphabet: Optional[str] = None, target_mode: str = "char",
                 transform: Optional[callable] = None, 
                 target_transform: Optional[callable] = None,
                 max_points: int = 512,
                 normalize: bool = True):
        self.root = Path(root)
        self.target_mode = target_mode
        self.transform = transform
        self.target_transform = target_transform
        self.max_points = max_points
        self.normalize = normalize
        
        # Load index
        with open(index_path, 'r') as f:
            self.samples = json.load(f)
        
        # Setup alphabet and character mapping
        if alphabet is None:
            self.alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()-_=+[]{}|,.<>?;:`\\"
        else:
            self.alphabet = alphabet
        
        # Create character to index mapping
        self.character_index_map = {c: i for i, c in enumerate(self.alphabet)}
        
        # Cache for parsed files (to avoid re-parsing)
        self._file_cache = {}
    
    def __len__(self):
        return len(self.samples)
    
    def _encode_label_char(self, label: str) -> int:
        """Encode a single character label to index."""
        if label and label.lower() in self.character_index_map:
            return self.character_index_map[label.lower()]
        return 0  # Default to first character if not found
    
    def _encode_label_sequence(self, label: str) -> torch.Tensor:
        """Encode a sequence of characters to tensor of indices."""
        index_list = [self.character_index_map.get(c.lower(), 0) for c in label]
        return torch.tensor(index_list, dtype=torch.int64)
    
    def _load_strokes(self, file_path: Path, segment_id: int) -> torch.Tensor:
        """Load and process strokes for a specific segment."""
        # Check cache
        if str(file_path) not in self._file_cache:
            self._file_cache[str(file_path)] = parse_unipen_file(file_path)
        
        parsed = self._file_cache[str(file_path)]
        strokes = parsed['strokes']
        
        # If we have segment information, extract relevant strokes
        if parsed['segments'] and segment_id < len(parsed['segments']):
            # For simplicity, use all strokes if segment info is available
            # You may want to refine this based on segment boundaries
            pass
        
        # Convert to tensor
        return strokes_to_tensor(strokes, self.max_points, self.normalize)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Union[int, torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (stroke_tensor, label) where:
            - stroke_tensor: Tensor of shape (max_points, 3) with [x, y, pen_up]
            - label: Integer index for "char" mode, or tensor of indices for "sequence" mode
        """
        sample = self.samples[index]
        file_path = self.root / sample["file_path"]
        segment_id = sample["segment_id"]
        label_str = sample["label"]
        
        # Load stroke data
        stroke_tensor = self._load_strokes(file_path, segment_id)
        
        # Apply transforms
        if self.transform:
            stroke_tensor = self.transform(stroke_tensor)
        
        # Encode label
        if self.target_mode == "char":
            label = self._encode_label_char(label_str)
        else:  # sequence mode
            label = self._encode_label_sequence(label_str)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return stroke_tensor, label
        