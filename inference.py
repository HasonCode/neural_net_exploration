"""
Inference script for trained handwriting recognition model.
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
from train_model import HandwritingLSTM, HandwritingGRU
from unipen_class import Unipen


def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', argparse.Namespace())
    
    # Get model type from checkpoint or default to lstm
    model_type = getattr(args, 'model_type', 'lstm')
    hidden_size = getattr(args, 'hidden_size', 128)
    num_layers = getattr(args, 'num_layers', 2)
    dropout = getattr(args, 'dropout', 0.3)
    num_classes = getattr(args, 'num_classes', 26)
    
    # Create model
    if model_type == 'lstm':
        model = HandwritingLSTM(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        model = HandwritingGRU(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def predict(model, strokes_tensor, alphabet, device):
    """Predict character from stroke tensor."""
    model.eval()
    with torch.no_grad():
        strokes_tensor = strokes_tensor.unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(strokes_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        confidence = probabilities[0][predicted].item()
        predicted_char = alphabet[predicted.item()]
    
    return predicted_char, confidence, probabilities[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Run inference on handwriting samples')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--root_dir', type=str, default='unipen/CDROM/train_r01_v07',
                        help='Root directory of UniPen dataset')
    parser.add_argument('--index_path', type=str, default='unipen_index.json',
                        help='Path to index JSON file')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of sample to predict')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to test')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, checkpoint = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Load dataset
    dataset = Unipen(
        root=args.root_dir,
        index_path=args.index_path,
        alphabet="abcdefghijklmnopqrstuvwxyz",
        target_mode="char",
        max_points=checkpoint.get('args', argparse.Namespace()).max_points if hasattr(checkpoint.get('args', argparse.Namespace()), 'max_points') else 512,
        normalize=True
    )
    
    alphabet = dataset.alphabet
    
    # Test on samples
    print(f"\nTesting on {args.num_samples} samples:")
    print("=" * 60)
    correct = 0
    
    for i in range(args.sample_idx, min(args.sample_idx + args.num_samples, len(dataset))):
        strokes, true_label = dataset[i]
        true_char = alphabet[true_label]
        
        predicted_char, confidence, probs = predict(model, strokes, alphabet, device)
        
        is_correct = predicted_char == true_char
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Sample {i}: True='{true_char}', Predicted='{predicted_char}' "
              f"(confidence: {confidence:.2%})")
    
    accuracy = 100 * correct / args.num_samples
    print("=" * 60)
    print(f"Accuracy: {correct}/{args.num_samples} ({accuracy:.2f}%)")


if __name__ == '__main__':
    main()

