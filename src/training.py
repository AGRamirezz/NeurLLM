"""
Training module for NeurLLM.

This module provides functions and classes for training and fine-tuning
transformer models on neurophysiological data.
"""

import os
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


class Trainer:
    """Trainer class for NeurLLM models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "./checkpoints",
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Device to train on
            max_grad_norm: Maximum norm for gradient clipping
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or AdamW(model.parameters(), lr=5e-5)
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        
        self.model.to(self.device)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, loss_fn: Callable) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            loss_fn: Loss function to use
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Training loop
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                inputs={k: v for k, v in batch.items() if k != "text_input" and k != "labels"},
                text_input=batch.get("text_input")
            )
            
            # Calculate loss
            loss = loss_fn(outputs, batch["labels"])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update parameters
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        return {"train_loss": total_loss / num_batches}
    
    def evaluate(self, loss_fn: Callable) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            loss_fn: Loss function to use
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    inputs={k: v for k, v in batch.items() if k != "text_input" and k != "labels"},
                    text_input=batch.get("text_input")
                )
                
                # Calculate loss
                loss = loss_fn(outputs, batch["labels"])
                
                # Log metrics
                total_loss += loss.item()
        
        return {"val_loss": total_loss / num_batches}
    
    def train(
        self,
        epochs: int,
        loss_fn: Callable,
        save_every: int = 1,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train for
            loss_fn: Loss function to use
            save_every: Save checkpoint every n epochs
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            Dictionary of metrics for each epoch
        """
        metrics = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(loss_fn)
            metrics["train_loss"].append(train_metrics["train_loss"])
            
            # Evaluate
            val_metrics = self.evaluate(loss_fn)
            metrics["val_loss"].append(val_metrics["val_loss"])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        
        return metrics
    
    def save_checkpoint(self, filename: str) -> str:
        """
        Save a checkpoint of the model.
        
        Args:
            filename: Name of the checkpoint file
            
        Returns:
            Path to the saved checkpoint
        """
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        return path
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint.
        
        Args:
            path: Path to the checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> AdamW:
    """
    Create an optimizer for the model.
    
    Args:
        model: The model
        lr: Learning rate
        weight_decay: Weight decay
        betas: Beta parameters for AdamW
        
    Returns:
        AdamW optimizer
    """
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas=betas)
    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
) -> LRScheduler:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: The optimizer
        num_training_steps: Total number of training steps
        warmup_ratio: Ratio of warmup steps
        
    Returns:
        Learning rate scheduler
    """
    from transformers import get_scheduler
    
    warmup_steps = int(num_training_steps * warmup_ratio)
    
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    return scheduler 