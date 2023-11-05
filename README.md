# Contribtuion
"""
This README file was generated using GitHub Copilot. It is intended for contribution purposes.
"""

# Deep Learning Weight Initialization Techniques

This repository contains implementations of proposed weight initialization techniques used in deep learning.

## Techniques

1. **Zero Initialization** - All weights are initialized to zero.
2. **Random Initialization** - Weights are initialized randomly.
3. **He Initialization** - Weights are initialized keeping in mind the size of the previous layer which helps in attaining a global minimum of the cost function faster and more accurately.
4. **Xavier/Glorot Initialization** - It's a way of initializing the weights such that the variance remains the same for x and y.

## Requirements

- Python 3.x
- NumPy


## Usage

Each technique is implemented in its own Python file. You can use them by importing the required file into your project.

```python
from weight_initializations import he_initialization

