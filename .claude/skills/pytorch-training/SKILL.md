---
name: pytorch-training
description: Use this skill whenever the task involves writing PyTorch code for the thesis project, including model architecture (CNN-LSTM, Transformer), training loops, fine-tuning, transfer learning, domain adaptation (DANN), or experiment configuration. Triggers include any mention of PyTorch, training loop, fine-tuning, freeze layers, learning rate scheduling, checkpointing, DANN, domain adaptation, or model architecture. Use this skill before writing any model or training code. Do NOT use for general PyTorch tutorials or unrelated PyTorch projects.
---

# PyTorch Training Skill

This skill covers patterns for writing PyTorch code for the maize yield prediction thesis project, with focus on transfer learning and reproducibility.

## Critical Constraints

- **Hardware target**: Code must run on Google Colab Pro (A100/V100/T4 GPU). Mac M1 only for testing with tiny dataset.
- **Reproducibility**: ALWAYS set seeds. Results that can't be reproduced are unpublishable.
- **Memory budget**: Colab Pro = 51GB RAM, ~16GB GPU. Don't load full datasets into RAM.
- **Wandb logging**: All training runs MUST be logged to wandb for thesis traceability.
- **Checkpointing**: Save every epoch. Colab can disconnect, losing hours of work.

## Standard Imports & Boilerplate

Every training script starts with this:

```python
import os
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import wandb
from tqdm import tqdm

# Reproducibility
def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device selection
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        # Mac M1 — only for testing
        return torch.device('mps')
    return torch.device('cpu')
```

## Model Architecture: You et al. (2017) CNN-LSTM Baseline

Reference architecture for crop yield prediction from histogram features.

```python
class HistogramCNNLSTM(nn.Module):
    """
    CNN-LSTM architecture from You et al. 2017.

    Input: (batch, n_timesteps, n_bands, n_bins)
           e.g., (32, 30, 9, 32)
    Output: (batch, 1) yield prediction
    """
    def __init__(
        self,
        n_bands: int = 9,
        n_bins: int = 32,
        n_timesteps: int = 30,
        cnn_channels: int = 128,
        lstm_hidden: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        # CNN: 2D conv over (bands, bins) per timestep
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # LSTM: temporal aggregation across timesteps
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, Bins)
        B, T, C, Bins = x.shape
        x = x.view(B * T, 1, C, Bins)
        cnn_out = self.cnn(x)  # (B*T, cnn_channels)
        cnn_out = cnn_out.view(B, T, -1)
        lstm_out, _ = self.lstm(cnn_out)
        last_hidden = lstm_out[:, -1, :]  # (B, lstm_hidden)
        return self.head(last_hidden)
```

## Training Loop Pattern

Standard training loop with wandb integration and checkpointing:

```python
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f'Epoch {epoch} [train]')
    for batch in pbar:
        x, y = batch['features'].to(device), batch['target'].to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    for batch in loader:
        x, y = batch['features'].to(device), batch['target'].to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())
    preds = np.concatenate(preds).flatten()
    targets = np.concatenate(targets).flatten()
    return {
        'loss': total_loss / len(loader),
        'rmse': np.sqrt(np.mean((preds - targets) ** 2)),
        'r2': 1 - np.sum((preds - targets) ** 2) / np.sum((targets - targets.mean()) ** 2),
        'mape': np.mean(np.abs((preds - targets) / targets)) * 100,
    }


def train_model(config: dict) -> None:
    """Main training entry point."""
    set_seeds(config['seed'])
    device = get_device()

    # Initialize wandb
    wandb.init(
        project='maize-transfer',
        name=config['run_name'],
        config=config,
        tags=config.get('tags', []),
    )

    # Build model, data, optimizer
    model = build_model(config).to(device)
    train_loader, val_loader = build_dataloaders(config)
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.MSELoss()

    # Train loop
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'lr': scheduler.get_last_lr()[0],
            **{f'val_{k}': v for k, v in val_metrics.items()},
        })

        # Checkpoint
        save_checkpoint(model, optimizer, epoch, config, val_metrics, is_best=val_metrics['loss'] < best_val_loss)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']

    wandb.finish()
```

## Transfer Learning Patterns

### Pattern 1: Vanilla Fine-Tuning

```python
def fine_tune(pretrained_path: str, target_config: dict) -> nn.Module:
    """Standard fine-tuning: load pretrained, replace head, retrain."""
    # Load pretrained
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    model = HistogramCNNLSTM(**target_config['model_kwargs'])
    model.load_state_dict(checkpoint['model_state'], strict=False)

    # Replace head with fresh weights
    model.head = nn.Sequential(
        nn.Linear(target_config['model_kwargs']['lstm_hidden'], 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 1),
    )

    return model
```

### Pattern 2: Frozen Feature Extractor

```python
def freeze_encoder(model: nn.Module) -> nn.Module:
    """Freeze CNN+LSTM, only train head."""
    for param in model.cnn.parameters():
        param.requires_grad = False
    for param in model.lstm.parameters():
        param.requires_grad = False
    return model
```

### Pattern 3: Layer-wise Learning Rate

```python
def get_layer_wise_optimizer(model: nn.Module, base_lr: float = 1e-4) -> Adam:
    """Lower LR for pretrained layers, higher for head."""
    return Adam([
        {'params': model.cnn.parameters(), 'lr': base_lr * 0.1},
        {'params': model.lstm.parameters(), 'lr': base_lr * 0.5},
        {'params': model.head.parameters(), 'lr': base_lr},
    ])
```

### Pattern 4: DANN (Domain Adversarial Neural Network)

```python
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DANNModel(nn.Module):
    """CNN-LSTM with domain classifier branch for adversarial domain adaptation."""
    def __init__(self, base_model: HistogramCNNLSTM, n_domains: int = 2):
        super().__init__()
        self.feature_extractor = nn.Sequential(base_model.cnn, base_model.lstm)
        self.yield_predictor = base_model.head
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_domains),
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        yield_pred = self.yield_predictor(features)
        reversed_features = GradientReversalFunction.apply(features, alpha)
        domain_pred = self.domain_classifier(reversed_features)
        return yield_pred, domain_pred
```

## Configuration File Format

All experiments are driven by YAML configs in `experiments/configs/`:

```yaml
# experiments/configs/usa_baseline.yaml
run_name: usa_baseline_v1
tags: [baseline, usa, from_scratch]
seed: 42

data:
  source_country: usa
  data_dir: data/processed/usa
  train_years: [2003, 2018]
  val_years: [2019, 2021]
  test_years: [2022, 2023]
  batch_size: 32
  num_workers: 4

model:
  type: HistogramCNNLSTM
  model_kwargs:
    n_bands: 9
    n_bins: 32
    n_timesteps: 30
    cnn_channels: 128
    lstm_hidden: 256
    dropout: 0.5

training:
  epochs: 100
  lr: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  early_stopping_patience: 15

checkpoint:
  save_dir: experiments/checkpoints/usa_baseline
  save_every: 1
```

## Checkpointing Pattern

```python
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    metrics: dict,
    is_best: bool,
) -> None:
    """Save checkpoint to disk and Drive (for Colab persistence)."""
    save_dir = Path(config['checkpoint']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': config,
        'metrics': metrics,
    }
    torch.save(state, save_dir / f'epoch_{epoch}.pt')
    if is_best:
        torch.save(state, save_dir / 'best.pt')
```

## Wandb Run Naming Convention

Use this template for ALL runs to enable filtering:
{source_country}{target_country}{method}{seed}{date}

Examples:
- `usa_usa_baseline_42_20260112` (USA from-scratch baseline)
- `usa_indonesia_finetune_42_20260205` (Vanilla fine-tune to Indonesia)
- `usa_indonesia_dann_42_20260220` (DANN domain adaptation to Indonesia)
- `usa_asean_finetune_42_20260301` (Fine-tune to all ASEAN combined)

## Common Pitfalls & Solutions

### Pitfall 1: Non-Reproducible Results
**Symptom**: Same code, different results each run.
**Solution**: Call `set_seeds()` at script start. Set `cudnn.deterministic=True`.

### Pitfall 2: Out-of-Memory on Colab
**Symptom**: CUDA OOM error.
**Solution**:
- Reduce batch size (try 16 or 8)
- Use gradient accumulation
- Use `torch.cuda.empty_cache()` between epochs
- Use mixed precision: `torch.cuda.amp.autocast()`

### Pitfall 3: Colab Disconnect Mid-Training
**Symptom**: Lose hours of training when Colab times out.
**Solution**:
- Save checkpoint every epoch
- Save to Google Drive, not local Colab disk
- Implement resume-from-checkpoint logic

### Pitfall 4: Negative Transfer
**Symptom**: Fine-tuned model performs WORSE than from-scratch on target.
**Solution**:
- Try freezing encoder (Pattern 2)
- Try lower learning rate for pretrained layers (Pattern 3)
- Try DANN (Pattern 4)
- Document this finding — it's still publishable

### Pitfall 5: Train/Val/Test Leak
**Symptom**: Suspiciously high val/test accuracy.
**Solution**: Split by **year**, not random. Train: 2003-2018, Val: 2019-2021, Test: 2022-2023.

## Validation Checklist Before Each Run

- [ ] Seeds set
- [ ] Wandb project name correct
- [ ] Run name follows convention
- [ ] Config file committed to git
- [ ] Checkpoint dir specified
- [ ] Train/val/test split is temporal (not random)
- [ ] Tested with 1 epoch first to verify no errors

## Related Files in Project

- Models: `src/models/cnn_lstm.py`, `src/models/dann.py`
- Training: `src/training/trainer.py`, `src/training/transfer.py`
- Configs: `experiments/configs/*.yaml`
- Utilities: `src/utils/checkpoint.py`, `src/utils/seed.py`

## When NOT to Use This Skill

- Pure data preprocessing (use `data-validation` skill)
- GEE operations (use `gee-extraction` skill)
- Paper writing (use `thesis-writing` skill)
- Tutorials/learning PyTorch basics (read PyTorch official tutorials instead)
