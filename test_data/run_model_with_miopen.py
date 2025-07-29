#!/usr/bin/env python3
"""
Simple PyTorch Model Runner with MIOpen Command Logging

Usage:
    MIOPEN_ENABLE_LOGGING_CMD=1 python3 run_model_with_miopen.py --model resnet 2> miopen_commands.txt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os

class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 1)  # 1x1 conv
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x

class MobileNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Depthwise (grouped) convolution
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch)
        # Pointwise convolution  
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        
    def forward(self, x):
        x = F.relu(self.depthwise(x))
        x = F.relu(self.pointwise(x))
        return x

class SimpleMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.block1 = MobileNetBlock(32, 64)
        self.block2 = MobileNetBlock(64, 128, stride=2)
        self.block3 = MobileNetBlock(128, 256, stride=2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, stride=stride)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(out_ch, out_ch * 4, 1)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.block1 = ResNetBlock(64, 64)
        self.block2 = ResNetBlock(256, 128, stride=2)
        self.block3 = ResNetBlock(512, 256, stride=2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simple', 'mobilenet', 'resnet'], default='simple')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--input-size', type=int, default=224)
    args = parser.parse_args()
    
    # Check MIOpen logging
    if not os.environ.get('MIOPEN_ENABLE_LOGGING_CMD'):
        print("WARNING: Set MIOPEN_ENABLE_LOGGING_CMD=1 to capture commands")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    if args.model == 'simple':
        model = SimpleConvNet()
    elif args.model == 'mobilenet':
        model = SimpleMobileNet()
    elif args.model == 'resnet':
        model = SimpleResNet()
    
    model = model.to(device)
    input_tensor = torch.randn(args.batch_size, 3, args.input_size, args.input_size).to(device)
    
    print(f"Running {args.model} model...")
    print(f"Input shape: {input_tensor.shape}")
    
    # Run inference (triggers MIOpen commands)
    model.eval()
    with torch.no_grad():
        for i in range(3):
            print(f"Iteration {i+1}")
            output = model(input_tensor)
            print(f"Output shape: {output.shape}")
    
    print("Done! MIOpen commands logged to stderr")

if __name__ == "__main__":
    main()