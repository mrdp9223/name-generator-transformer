# Sanskrit Name Generator

A deep learning model that generates Sanskrit names using a Transformer-based architecture. This project demonstrates the application of Transformer models to character-level language generation, showcasing how self-attention mechanisms can capture intricate patterns in Sanskrit names.

## Features

- Transformer-based neural network architecture
- Character-level name generation
- Unicode normalization for Sanskrit text
- Temperature-controlled sampling for diverse outputs
- Early stopping and validation to prevent overfitting
- Handles Sanskrit diacritical marks appropriately



## Requirements

- Python 3.7+
- PyTorch
- Unicode support for Sanskrit text


```

Install the required packages:
```bash
py -m pip install -r requirements.txt
```

## Usage

1. Prepare your training data:
   - Create a text file named `sanskrit_names.txt`
   - Add one Sanskrit name per line
   - Save in UTF-8 encoding

2. Train the model:
```bash
python generate_names.py
```

3. The script will:
   - Load and preprocess the names
   - Train the model
   - Generate example names

## Model Architecture

The name generator uses a Transformer-based architecture with:
- Embedding dimension: 32
- Number of attention heads: 4
- Number of transformer layers: 2
- Feedforward dimension: 64
- Context window size: 10 characters

## Training Parameters

- Batch size: 16
- Maximum epochs: 50
- Learning rate: 0.005
- Early stopping patience: 5 epochs
- Train/Validation split: 80/20

