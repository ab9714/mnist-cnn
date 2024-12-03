# MNIST CNN Assignment

## Total Parameter Count
Total parameters: [9410]
 Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 28, 28]              40
       BatchNorm2d-2            [-1, 4, 28, 28]               8
         MaxPool2d-3            [-1, 4, 14, 14]               0
            Conv2d-4            [-1, 8, 14, 14]             296
       BatchNorm2d-5            [-1, 8, 14, 14]              16
         MaxPool2d-6              [-1, 8, 7, 7]               0
            Conv2d-7             [-1, 16, 7, 7]           1,168
       BatchNorm2d-8             [-1, 16, 7, 7]              32
           Dropout-9                  [-1, 784]               0
           Linear-10                   [-1, 10]           7,850
================================================================
Total params: 9,410
Trainable params: 9,410
## Use of Batch Normalization
Yes, Batch Normalization is used in the model.

## Use of DropOut
Yes, DropOut is used in the model.

## Use of Fully Connected Layer
Yes, a Fully Connected Layer is used in the model.

## Validation/Test Accuracy
Test accuracy: [98%]

## Links
- [GitHub Repository](https://github.com/ab9714/mnist-cnn)