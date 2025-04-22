"""
Evaluation module for NeurLLM.

This module provides tools for evaluating models and visualizing 
neurophysiological data and model predictions.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def compute_classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        labels: Ground truth labels
        predictions: Model predictions
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_regression_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        labels: Ground truth values
        predictions: Model predictions
        
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def compute_perplexity(model: torch.nn.Module, dataloader: DataLoader, device: str) -> float:
    """
    Compute perplexity on a dataset.
    
    Args:
        model: The model
        dataloader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                inputs={k: v for k, v in batch.items() if k != "text_input" and k != "labels"},
                text_input=batch.get("text_input")
            )
            
            # Calculate loss using CrossEntropyLoss
            loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), batch["labels"].view(-1))
            
            # Count tokens (excluding padding tokens)
            tokens = (batch["labels"] != -100).sum().item()
            
            total_loss += loss.item()
            total_tokens += tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


class ModelAnalyzer:
    """Class for analyzing model behavior and visualizing results."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./analysis"
    ):
        """
        Initialize the analyzer.
        
        Args:
            model: The model to analyze
            device: Device to run analysis on
            output_dir: Directory to save analysis results
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        
        self.model.to(self.device)
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_attention(
        self,
        inputs: Dict[str, torch.Tensor],
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
        modality: str = "eeg",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize attention patterns.
        
        Args:
            inputs: Input tensors
            layer_idx: Index of the layer to visualize
            head_idx: Index of the attention head (None for all heads)
            modality: Modality to visualize
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        self.model.eval()
        
        # Get attention weights
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # This assumes the model has a method to get attention weights
        # You might need to modify the model to store and return attention weights
        with torch.no_grad():
            outputs = self.model(inputs)
            attention = self.model.get_attention_weights()  # This is a placeholder method
        
        # Get the attention weights for the specified layer
        layer_attention = attention[layer_idx]
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if head_idx is not None:
            # Visualize a single head
            attn = layer_attention[0, head_idx].cpu().numpy()
            sns.heatmap(attn, cmap="viridis", ax=ax)
            ax.set_title(f"Attention weights for layer {layer_idx}, head {head_idx}")
        else:
            # Visualize average over all heads
            attn = layer_attention[0].mean(dim=0).cpu().numpy()
            sns.heatmap(attn, cmap="viridis", ax=ax)
            ax.set_title(f"Average attention weights for layer {layer_idx}")
        
        ax.set_xlabel("Key positions")
        ax.set_ylabel("Query positions")
        
        # Save the figure
        if save_path:
            fig.savefig(save_path)
        
        return fig
    
    def visualize_modality_embeddings(
        self,
        dataloader: DataLoader,
        num_samples: int = 100,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize embeddings for different modalities using dimensionality reduction.
        
        Args:
            dataloader: DataLoader for test data
            num_samples: Number of samples to visualize
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        from sklearn.decomposition import PCA
        
        self.model.eval()
        
        # Collect embeddings
        embeddings = []
        modalities = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i * dataloader.batch_size >= num_samples:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get embeddings for each modality
                # This assumes the model has a method to get embeddings
                # You might need to modify the model to store and return embeddings
                modality_embeddings = self.model.get_modality_embeddings(batch)  # Placeholder
                
                for modality, emb in modality_embeddings.items():
                    embeddings.append(emb.cpu().numpy())
                    modalities.extend([modality] * emb.shape[0])
        
        # Concatenate all embeddings
        embeddings = np.concatenate(embeddings, axis=0)[:num_samples]
        modalities = modalities[:num_samples]
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each modality with a different color
        unique_modalities = set(modalities)
        for modality in unique_modalities:
            indices = [i for i, m in enumerate(modalities) if m == modality]
            ax.scatter(
                reduced_embeddings[indices, 0],
                reduced_embeddings[indices, 1],
                label=modality,
                alpha=0.7
            )
        
        ax.set_title("Modality Embeddings Visualization")
        ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%})")
        ax.legend()
        
        # Save the figure
        if save_path:
            fig.savefig(save_path)
        
        return fig


def visualize_time_series(
    data: np.ndarray,
    sampling_rate: float,
    channels: Optional[List[str]] = None,
    title: str = "Time Series Data",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize neurophysiological time series data.
    
    Args:
        data: Time series data of shape (channels, time_steps)
        sampling_rate: Sampling rate in Hz
        channels: List of channel names
        title: Title of the figure
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    num_channels, time_steps = data.shape
    time = np.arange(time_steps) / sampling_rate
    
    if channels is None:
        channels = [f"Channel {i+1}" for i in range(num_channels)]
    
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels), sharex=True)
    
    if num_channels == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(time, data[i], linewidth=0.8)
        ax.set_ylabel(channels[i])
        ax.grid(True)
    
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    return fig 