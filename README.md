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


# Training Logs
Total number of parameters: 13808

EPOCH: 1/20
Loss=0.026055309921503067 Batch_id=937 Accuracy=93.31: 10 

Test set: Average loss: 0.0528, Accuracy: 9832/10000 (98.32%)


EPOCH: 2/20
Loss=0.011358045972883701 Batch_id=937 Accuracy=97.97: 10 

Test set: Average loss: 0.0362, Accuracy: 9888/10000 (98.88%)


EPOCH: 3/20
Loss=0.11771378666162491 Batch_id=937 Accuracy=98.48: 100 

Test set: Average loss: 0.0314, Accuracy: 9912/10000 (99.12%)


EPOCH: 4/20
Loss=0.004047331400215626 Batch_id=937 Accuracy=98.60: 10 

Test set: Average loss: 0.0246, Accuracy: 9923/10000 (99.23%)


EPOCH: 5/20
Loss=0.0036711778957396746 Batch_id=937 Accuracy=98.78: 1 

Test set: Average loss: 0.0233, Accuracy: 9923/10000 (99.23%)


EPOCH: 6/20
Loss=0.006688658148050308 Batch_id=937 Accuracy=98.86: 10 

Test set: Average loss: 0.0239, Accuracy: 9921/10000 (99.21%)


EPOCH: 7/20
Loss=0.007131639402359724 Batch_id=937 Accuracy=98.89: 10 

Test set: Average loss: 0.0258, Accuracy: 9919/10000 (99.19%)


EPOCH: 8/20
Loss=0.01137285865843296 Batch_id=937 Accuracy=99.00: 100 

Test set: Average loss: 0.0245, Accuracy: 9927/10000 (99.27%)


EPOCH: 9/20
Loss=0.0019478006288409233 Batch_id=937 Accuracy=99.04: 1 

Test set: Average loss: 0.0235, Accuracy: 9929/10000 (99.29%)


EPOCH: 10/20
Loss=0.004158478695899248 Batch_id=937 Accuracy=99.05: 10 

Test set: Average loss: 0.0221, Accuracy: 9931/10000 (99.31%)


EPOCH: 11/20
Loss=0.0027304713148623705 Batch_id=937 Accuracy=99.07: 1 

Test set: Average loss: 0.0184, Accuracy: 9941/10000 (99.41%)


EPOCH: 12/20
Loss=0.08624891936779022 Batch_id=937 Accuracy=99.14: 100 

Test set: Average loss: 0.0210, Accuracy: 9931/10000 (99.31%)


EPOCH: 13/20
Loss=0.024660026654601097 Batch_id=937 Accuracy=99.16: 10 

Test set: Average loss: 0.0197, Accuracy: 9941/10000 (99.41%)


EPOCH: 14/20
Loss=0.002963076811283827 Batch_id=937 Accuracy=99.22: 10 

Test set: Average loss: 0.0183, Accuracy: 9943/10000 (99.43%)


EPOCH: 15/20
Loss=0.11881911009550095 Batch_id=937 Accuracy=99.17: 100 

Test set: Average loss: 0.0181, Accuracy: 9944/10000 (99.44%)


EPOCH: 16/20
Loss=0.06918376684188843 Batch_id=937 Accuracy=99.26: 100 

Test set: Average loss: 0.0204, Accuracy: 9938/10000 (99.38%)


EPOCH: 17/20
Loss=0.0030055332463234663 Batch_id=937 Accuracy=99.20: 1 

Test set: Average loss: 0.0190, Accuracy: 9939/10000 (99.39%)


EPOCH: 18/20
Loss=0.027890503406524658 Batch_id=937 Accuracy=99.24: 10 

Test set: Average loss: 0.0209, Accuracy: 9938/10000 (99.38%)


EPOCH: 19/20
Loss=0.01147868949919939 Batch_id=937 Accuracy=99.31: 100 

Test set: Average loss: 0.0181, Accuracy: 9940/10000 (99.40%)


EPOCH: 20/20
Loss=0.0037440010346472263 Batch_id=937 Accuracy=99.31: 1 

Test set: Average loss: 0.0186, Accuracy: 9941/10000 (99.41%)

## Requirements

To run this code, you need to have the following libraries installed:

- `torch`
- `torchvision`
- `tqdm`
- `torchsummary`

You can install the required libraries using pip:

## Links
- [GitHub Repository](https://github.com/ab9714/mnist-cnn)