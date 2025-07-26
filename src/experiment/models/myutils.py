import torch.nn as nn


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False

