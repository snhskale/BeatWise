import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# same architecture we used for training.

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class ECG_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ECG_ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride_val in strides:
            layers.append(block(self.in_channels, out_channels, stride_val))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def classify_heartbeats(processed_data, model_path='final_model.pth'):
    """
    The main function of the agent. Takes preprocessed heartbeat data
    and returns an array of classified labels using the final model.
    """
    # Define the exact classes the model was trained on
    main_classes = ['/', 'A', 'L', 'N', 'R', 'V']
    label_encoder = LabelEncoder().fit(main_classes)
    num_classes = len(main_classes)
    
    # Set the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We use a ResNet18-like structure, adapted for 1D
    model = ECG_ResNet(ResidualBlock, [2, 2, 2], num_classes=num_classes).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        return None
    
    model.eval() # Set the model to evaluation mode

    # Prepare the data for prediction
    input_tensor = torch.tensor(processed_data, dtype=torch.float32).to(device)

    # Make the predictions
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_indices = torch.max(outputs.data, 1)
    
    # Decode the predictions from numbers back to labels
    predicted_labels = label_encoder.inverse_transform(predicted_indices.cpu().numpy())
    
    return predicted_labels

