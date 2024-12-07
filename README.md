# MNIST CNN Model

This repository contains a Convolutional Neural Network (CNN) model implemented in PyTorch for classifying handwritten digits from the MNIST dataset.

## Model Architecture

The CNN model consists of several convolutional blocks, each followed by ReLU activation, Batch Normalization, and Dropout layers. The architecture is designed to improve accuracy while maintaining a manageable number of parameters.

### Key Features
- **Dropout**: Used to prevent overfitting.
- **Batch Normalization**: Used to stabilize and accelerate training.

## Training Parameters

- **Learning Rate**: 0.01
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum of 0.9
- **Batch Size**: 64
- **Number of Epochs**: 20
- **Dropout Rate**: 0.1

## Use of Dropout and Batch Normalization

- **Dropout**: `True`
- **Batch Normalization**: `True`

## Best Accuracy

The model achieved a best accuracy of approximately **99.46%** on the validation set.

## Requirements

To run this code, you need to have the following libraries installed:

- `torch`
- `torchvision`
- `tqdm`
- `torchsummary`

You can install the required libraries using pip:

## Links
- [GitHub Repository](https://github.com/ab9714/mnist-cnn)