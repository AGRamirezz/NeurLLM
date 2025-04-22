# NeurLLM: Vision Transformer for Neurophysiological Data

This notebook demonstrates how to fine-tune a Vision Transformer (ViT) encoder-decoder architecture on neurophysiological data (EEG) converted to spectrograms.

## Overview

This demonstration includes:

1. Converting EEG time series data to spectrograms/PSD images
2. Training a ViT encoder-decoder model for both classification and reconstruction
3. Analyzing the model's performance and latent space representations
4. Exploring applications in driving behavior classification

## Key Features

- **Data Generation**: Creates synthetic EEG data with class-specific frequency characteristics
- **Spectrogram Conversion**: Transforms EEG time series into visual representations
- **ViT Architecture**: Implements a dual-purpose encoder-decoder model for both classification and reconstruction
- **Interactive Demo**: Includes an interactive interface for exploring model predictions and attention maps
- **Latent Space Analysis**: Visualizes the learned representations using PCA and t-SNE
- **Interpretability**: Demonstrates attention map visualization for model interpretability

## Driving Behaviors Classification

The notebook focuses on classifying five driving behaviors:
- Smooth Driving
- Acceleration
- Deceleration
- Lane Change
- Turning

## Requirements

The notebook requires the following packages:
```
torch torchvision torchaudio timm einops scipy scikit-learn matplotlib seaborn ipywidgets mne
```

## Usage

1. Run the notebook cells sequentially to set up the environment and generate synthetic data
2. Visualize the raw EEG data and the corresponding spectrograms
3. Train the ViT encoder-decoder model
4. Evaluate the model and explore the interactive demo
5. Analyze the latent space and attention mechanisms

## Model Architecture

The implemented `NeurLLM_EncoderDecoder` model features:
- A pre-trained ViT encoder
- A custom transformer decoder
- A classification head for driving behavior identification
- A reconstruction capability for generating spectrogram images from latent representations 
