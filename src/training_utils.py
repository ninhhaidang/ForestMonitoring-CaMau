"""
Training Utilities
Các hàm hỗ trợ training process
"""

import torch
from pathlib import Path

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
    return model, optimizer, epoch, loss
