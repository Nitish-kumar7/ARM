import torch
import torch.nn as nn
from torchvision.models import resnet18
import numpy as np
import cv2
import os

class ARMClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ARMClassifier, self).__init__()
        # Load pretrained ResNet18
        self.model = resnet18(pretrained=True)
        
        # Modify the final layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Store the last convolutional layer for Grad-CAM
        self.target_layer = self.model.layer4[-1]
        
        # Grad-CAM variables
        self.gradients = None
        self.activations = None
        
        # Register hooks for Grad-CAM
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def forward(self, x):
        return self.model(x)
    
    def get_gradcam(self, x, index=None):
        """Generate Grad-CAM heatmap for the input image"""
        # Forward pass
        output = self.forward(x)
        
        if index is None:
            index = output.argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        output[0][index].backward()
        
        # Get weights
        weights = torch.mean(self.gradients, dim=(2, 3))
        
        # Create heatmap
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)
        
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i, :, :]
        
        # Apply ReLU and normalize
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().numpy()
    
    def predict_with_gradcam(self, x):
        """Get prediction and Grad-CAM heatmap for an input image"""
        # Get prediction
        with torch.no_grad():
            output = self.forward(x)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Get Grad-CAM heatmap
        heatmap = self.get_gradcam(x, prediction)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'heatmap': heatmap
        }

def load_model(model_path=None):
    """Load the model from a saved state"""
    model = ARMClassifier()
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.eval()
    return model 