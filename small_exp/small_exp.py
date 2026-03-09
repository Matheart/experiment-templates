"""
Small-scale MLP training for experimentation.
A simple two-layer multilayer perceptron with configurable hyperparameters.

lr sweep: 1e-3, 1e-4, 1e-5
wd sweep: 1e-3, 1e-4, 1e-5, 1e-6
batch size sweep: 32, 64, 128, 256, 512
width sweep: 64, 128, 256, 512, 1024, 2048
Total: 3 * 4 * 5 * 6 = 360 experiments

uv run small_mlp.py --lr 1e-3 --batch-size 32 --wd 1e-5 --width 512
uv run small_mlp.py --lr 1e-4 --batch-size 32 --wd 1e-5 --width 512 
uv run small_mlp.py --lr 1e-5 --batch-size 32 --wd 1e-5 --width 512 
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import random
import numpy as np


class MLP(nn.Module):
    """Two-layer multilayer perceptron."""
    
    def __init__(self, input_dim: int, width: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train(model, train_loader, epochs: int, criterion, optimizer, device):
    """Train the model for multiple epochs."""
    for epoch in range(epochs):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Small-scale MLP training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--width', type=int, default=512, help='Hidden layer width')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--input-dim', type=int, default=10, help='Input dimension')
    parser.add_argument('--output-dim', type=int, default=50000, help='Output dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, seed: {args.seed}")

    start_time = time.time()
    
    # Generate dummy data for demonstration
    # In practice, replace this with your actual dataset
    train_x = torch.randn(1000, args.input_dim)
    train_y = torch.randint(0, args.output_dim, (1000,))
    
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # Initialize model, loss, and optimizer
    model = MLP(args.input_dim, args.width, args.output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Starting training with lr={args.lr}, batch_size={args.batch_size}, "
          f"wd={args.wd}, width={args.width}")
    
    # Train
    train(model, train_loader, args.epochs, criterion, optimizer, device)
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
