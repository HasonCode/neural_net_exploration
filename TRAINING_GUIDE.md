# Handwriting Recognition Training Guide

This guide explains the best approach to train a handwriting recognition model using the UniPen stroke data.

## Overview

Your data consists of **stroke sequences** - temporal sequences of (x, y, pen_up) coordinates. This is fundamentally different from image-based handwriting recognition, which makes it well-suited for **recurrent neural networks (RNNs)**.

## Recommended Approach

### 1. **LSTM/GRU Models (Recommended Starting Point)**

**Why LSTM/GRU?**
- Stroke data is inherently sequential/temporal
- LSTMs excel at learning long-term dependencies in sequences
- They naturally handle variable-length sequences (with padding)
- Proven effective for handwriting recognition from strokes

**Architecture:**
- Input: `(batch_size, max_points, 3)` where 3 = [x, y, pen_up]
- LSTM/GRU layers to process the sequence
- Fully connected layers for classification
- Output: Logits for 26 character classes

### 2. **Alternative: Transformer Models**

For state-of-the-art results, consider Transformer architectures:
- Self-attention mechanisms capture relationships between stroke points
- Can handle longer sequences better
- More parameters, requires more data and compute

### 3. **Hybrid: CNN + RNN**

If you convert strokes to images:
- Use CNN to extract spatial features
- Use RNN to process temporal sequence
- More complex but can capture both spatial and temporal patterns

## Quick Start

### Step 1: Install Dependencies

```bash
pip install torch torchvision pillow numpy tqdm
```

### Step 2: Build Index (if not done)

```bash
python example_usage.py
```

Or manually:
```python
from unipen_class import build_index
build_index("unipen/CDROM/train_r01_v07", "unipen_index.json")
```

### Step 3: Train the Model

**Basic training:**
```bash
python train_model.py
```

**With custom parameters:**
```bash
python train_model.py \
    --model_type lstm \
    --hidden_size 256 \
    --num_layers 3 \
    --batch_size 64 \
    --epochs 30 \
    --lr 0.001 \
    --max_points 512
```

**Using GRU (faster, similar performance):**
```bash
python train_model.py --model_type gru
```

### Step 4: Test the Model

```bash
python inference.py --checkpoint checkpoints/best.pth --num_samples 20
```

## Model Architecture Details

### HandwritingLSTM

```
Input (batch, 512, 3)
    ↓
LSTM (2 layers, 128 hidden, bidirectional)
    ↓
Last timestep output (batch, 256)
    ↓
FC1 (256 → 128) + ReLU + Dropout
    ↓
FC2 (128 → 26)
    ↓
Output logits (batch, 26)
```

### Key Hyperparameters

- **hidden_size**: 128-256 (larger = more capacity, slower)
- **num_layers**: 2-3 (deeper = more complex patterns, risk of overfitting)
- **dropout**: 0.3-0.5 (regularization)
- **max_points**: 512 (should cover most characters)
- **batch_size**: 32-64 (depends on GPU memory)
- **learning_rate**: 0.001 (start here, adjust with scheduler)

## Training Tips

### 1. **Data Preprocessing**
- ✅ Normalization is already handled (coordinates to [0,1])
- ✅ Padding/truncation is handled
- Consider data augmentation:
  - Random noise to coordinates
  - Slight rotation/scaling
  - Temporal jittering

### 2. **Training Strategy**
- Start with smaller model (hidden_size=128, num_layers=2)
- Use validation split (20% default)
- Monitor validation accuracy to avoid overfitting
- Use learning rate scheduling (already included)
- Early stopping if validation doesn't improve

### 3. **Handling Class Imbalance**
If some characters are rare:
- Use weighted loss: `nn.CrossEntropyLoss(weight=class_weights)`
- Oversample rare classes
- Use focal loss

### 4. **Improving Performance**
- **More data**: Use data augmentation
- **Larger model**: Increase hidden_size and num_layers
- **Longer training**: More epochs with patience
- **Better architecture**: Try Transformer or attention mechanisms
- **Ensemble**: Train multiple models and average predictions

## Expected Results

With the default LSTM model:
- **Training accuracy**: 85-95% (after 20 epochs)
- **Validation accuracy**: 75-85% (depends on dataset size)
- **Training time**: ~10-30 minutes on CPU, ~2-5 minutes on GPU

## Advanced Techniques

### 1. **Attention Mechanism**
Add attention to focus on important parts of the stroke sequence:
```python
class AttentionLSTM(nn.Module):
    # Add attention layer after LSTM
    # Helps model focus on relevant stroke segments
```

### 2. **Sequence-to-Sequence**
For multi-character recognition:
- Use encoder-decoder architecture
- CTC loss for alignment-free training
- Beam search for decoding

### 3. **Transfer Learning**
- Pre-train on larger handwriting datasets
- Fine-tune on UniPen data
- Use pre-trained word embeddings

### 4. **Multi-task Learning**
Train on multiple tasks simultaneously:
- Character classification
- Writer identification
- Stroke prediction

## Troubleshooting

### Low Accuracy
- Check data quality and labels
- Increase model capacity
- Train longer
- Add regularization (dropout)
- Check for class imbalance

### Overfitting
- Increase dropout
- Reduce model size
- Add more data augmentation
- Use early stopping

### Slow Training
- Reduce batch size
- Reduce max_points
- Use GRU instead of LSTM
- Enable GPU if available

### Out of Memory
- Reduce batch_size
- Reduce max_points
- Reduce hidden_size
- Use gradient accumulation

## Next Steps

1. **Start simple**: Train with default parameters
2. **Monitor metrics**: Watch train/val loss and accuracy
3. **Iterate**: Adjust hyperparameters based on results
4. **Experiment**: Try different architectures
5. **Deploy**: Use inference script for predictions

## Files Overview

- `train_model.py`: Main training script with LSTM/GRU models
- `inference.py`: Script to test trained models
- `unipen_class.py`: Dataset class and data loading
- `example_usage.py`: Basic data loading example

## References

- LSTM for handwriting: Graves & Schmidhuber (2009)
- Stroke-based recognition: Liwicki et al. (2007)
- Transformer for sequences: Vaswani et al. (2017)

