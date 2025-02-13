# Neural Network Implementation with Full and Half Precision

This repository contains implementations of a neural network architecture in both full precision (float32) and half precision (float16) formats.

## Requirements

```
python >= 3.8
torch >= 2.0.0
pathlib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/emmsam/mixed_precision_training.git
cd mixed_precision_training
```

2. Install dependencies:
```bash
pip install torch pathlib
```

## Project Structure

- `network.py`: Contains the full-precision implementation
  - `BigNet`: Main network architecture
  - `LayerNorm`: Custom layer normalization implementation
  - `BIGNET_DIM`: Network dimension constant (default: 1024)

- `half_precision.py`: Contains the half-precision implementation
  - `HalfLinear`: Custom half-precision linear layer
  - `HalfBigNet`: Half-precision version of the network
  - Inherits LayerNorm from bignet.py

## Usage

### Loading and Using the Models

```python
from pathlib import Path
from bignet import load as load_full
from half_bignet import load as load_half

# Load full precision model
full_model = load_full(None)  # Without pretrained weights
# OR
full_model = load_full(Path("path/to/weights.pth"))  # With pretrained weights

# Load half precision model
half_model = load_half(None)  # Without pretrained weights
# OR
half_model = load_half(Path("path/to/weights.pth"))  # With pretrained weights

# Using the models
import torch

# Create input tensor (batch_size, BIGNET_DIM)
x = torch.randn(1, 1024)  # Example input

# Get outputs
full_output = full_model(x)
half_output = half_model(x)
```

### Memory Usage Comparison

The half-precision model (`HalfBigNet`) uses approximately half the memory of the full-precision model (`BigNet`) for its weights, while maintaining float32 compatibility for inputs and outputs.

### Key Features

1. **Full Precision Model**:
   - All weights and computations in float32
   - Standard PyTorch linear layers
   - Custom layer normalization

2. **Half Precision Model**:
   - Linear layer weights stored in float16
   - Computations performed in half precision
   - Layer normalization remains in full precision for stability
   - Input/output interface remains float32 compatible

## Notes

- The half-precision model automatically handles dtype conversion internally
- Layer normalization is kept in full precision to maintain numerical stability
- Both models use the same architecture with residual connections
- Default dimension (BIGNET_DIM) is set to 1024

## Memory Requirements

- Full precision model: ~4GB
- Half precision model: ~2GB