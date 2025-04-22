# NeurLLM

A project for building and scaling the next generation of brain foundation models. This repository contains the infrastructure for training large-scale transformer architectures that process continuous, multi-dimensional neural and behavioral time series data.

## Project Overview

The NeurLLM project aims to develop transformer-based models that can understand and generate predictions from neurophysiological data. The system processes multimodal inputs including:

- EEG recordings
- fMRI data
- Neural spike recordings
- Behavioral measurements
- Other neurophysiological signals

## Repository Structure

- `src/` - Core source code
  - `models.py` - Core model architectures (NeuroEncoder, MultimodalFusion, NeurLLM)
  - `data.py` - Data loading and preprocessing utilities
  - `training.py` - Training loops and optimization
  - `evaluation.py` - Evaluation metrics and visualization
- `Demo_Exercise_1/` - Vision Transformer for EEG spectrogram classification
- `Demo_Exercise_2/` - *Coming soon*
- `Demo_Exercise_3/` - *Coming soon*

## Demo Exercises

### Demo Exercise 1: Vision Transformer for Neurophysiological Data

The first demo exercise (`NeurLLM_Finetune_demo1.ipynb`) demonstrates how to:
- Convert EEG time series to spectrograms/PSD images
- Train a ViT encoder-decoder model for classification and reconstruction
- Analyze model performance and latent space representations
- Explore applications in driving behavior classification

### Planned Future Demos

- **Demo Exercise 2**: Multimodal fusion of EEG and behavioral data
- **Demo Exercise 3**: Generative modeling of neurophysiological signals
- **Demo Exercise 4**: Fine-tuning for specific clinical applications

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neurllm.git
cd neurllm

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## TODO Items

### Core Framework
- [ ] Complete the implementation of data loaders for additional neurophysiological data formats
- [ ] Expand model architecture to support more complex multimodal fusion
- [ ] Implement distributed training support for larger models
- [ ] Add support for quantization and model optimization

### Demo Exercises
- [ ] Create Demo Exercise 2: Multimodal fusion architecture
- [ ] Create Demo Exercise 3: Generative modeling with neurophysiological data
- [ ] Create Demo Exercise 4: Clinical applications and transfer learning
- [ ] Add documentation for each demo exercise

## Requirements

See `requirements.txt` for a list of dependencies. 