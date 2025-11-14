import numpy as np
import torch
from train_model import CNN

def load_mouse_data(points_array, max_points = 512):
    points  = np.array(points_array,dtype=np.float32)
    if len(points)>max_points:
        points = points[:max_points]
    else:
        padding_zeros = np.zeros((max_points-len(points),3),dtype=np.float32)
        points = np.concatenate([points,padding_zeros],axis =0)
    return torch.from_numpy(points)

def predict_from_points(model_path, points_array, num_classes, alphabet=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(input_channels=3,num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.to(device)
    model.eval()

    stroke_tensor = load_mouse_data(points_array)
    stroke_tensor = stroke_tensor.permute(1,0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(stroke_tensor)
        probabilities = torch.softmax(stroke_tensor,dim=1)
        predicted_idx = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()

    if alphabet:
        predicted_char = alphabet[predicted_idx]
        return predicted_char, confidence
    else:
        return predicted_idx,confidence

