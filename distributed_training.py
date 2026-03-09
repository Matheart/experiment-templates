"""
Distributed training for larger-scale MLP using PyTorch DistributedDataParallel.
Supports multi-GPU training across multiple nodes.
"""

import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


class LargeMLP(nn.Module):
    """Large-scale multilayer perceptron for distributed training."""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def setup(rank: int, world_size: int):
    """Initialize the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def train_epoch(model, dataloader, criterion, optimizer, rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(rank)
        batch_y = batch_y.to(rank)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def train(model, train_loader, epochs: int, criterion, optimizer, rank):
    """Train the model for multiple epochs."""
    for epoch in range(epochs):
        # Set epoch for distributed sampler to ensure proper shuffling
        train_loader.sampler.set_epoch(epoch)
        
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, rank)
        
        if (epoch + 1) % 10 == 0 and rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


def run_distributed(rank: int, world_size: int, args):
    """Run distributed training on a single process."""
    # Initialize distributed environment
    setup(rank, world_size)
    
    # Initialize model and move to correct device
    model = LargeMLP(
        args.input_dim,
        [args.width] * args.layers,
        args.output_dim
    ).to(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # Generate dummy data for demonstration
    # In practice, replace this with your actual dataset
    train_x = torch.randn(args.data_size, args.input_dim)
    train_y = torch.randint(0, args.output_dim, (args.data_size,))
    
    train_dataset = TensorDataset(train_x, train_y)
    
    # Use DistributedSampler to partition data across processes
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Starting training with lr={args.lr}, batch_size={args.batch_size}, "
              f"wd={args.wd}, width={args.width}, layers={args.layers}")
    
    # Train
    train(model, train_loader, args.epochs, criterion, optimizer, rank)
    
    if rank == 0:
        print("Training completed!")
    
    cleanup()


def main():
    parser = argparse.ArgumentParser(description='Distributed MLP training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--width', type=int, default=1024, help='Hidden layer width')
    parser.add_argument('--layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--input-dim', type=int, default=784, help='Input dimension')
    parser.add_argument('--output-dim', type=int, default=10, help='Output dimension')
    parser.add_argument('--data-size', type=int, default=10000, help='Dataset size')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs per node')
    
    args = parser.parse_args()
    
    world_size = args.nodes * args.gpus
    
    # Use multiprocessing spawn to start multiple processes
    mp.spawn(
        run_distributed,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
