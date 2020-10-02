import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_input_classes, n_output_classes, conv1_channels, conv2_channels,reg):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        stride = 1
        conv1_channels, int - number of filters in the 1st conv layer, padding=1
        after first conv  - shape (2, 32, 32, 2)
        after first max pooling  - shape (2, 8, 8, 2)
        conv2_channels, int - number of filters in the 2nd conv layer, padding=1
        after second conv  - shape (2, 8, 8, 2)
        after second max poolling - width =2, height = 2 channels = 2
        for convolutional layer in_channels = 8
        """
        # TODO Create necessary layers
        layers = []
        image_width, image_height, channels = input_shape
        conv1_layer = ConvolutionalLayer(in_channels=channels, out_channels=conv1_channels, filter_size=3, padding=1)
        layers.append(conv1_layer)
        layers.append(ReLULayer())
        layers.append(MaxPoolingLayer(4,4))
        
        
        conv2_layer = ConvolutionalLayer(in_channels=conv1_channels, out_channels=conv2_channels, filter_size=3, padding=1)
        
        layers.append(conv2_layer)
        layers.append(ReLULayer())
        layers.append(MaxPoolingLayer(4,4))
        layers.append(Flattener())
        layers.append(FullyConnectedLayer(n_input_classes, n_output_classes))
        
        self.layers = layers
        self.reg = reg
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        temp = X
        # zero graidents
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
            
        
        for layer in self.layers:
            temp = layer.forward(temp)

        # calculate loss
        loss, dprediction = softmax_with_cross_entropy(temp, y)

    
        d_current = dprediction
        for layer in reversed(self.layers):
            d_current = layer.backward(d_current)
            
        for _, param in self.params().items():
            loss_p, grad_p = l2_regularization(param.value, self.reg)
            loss += loss_p
            param.grad += grad_p

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        temp = X
        for layer in self.layers:
            temp = layer.forward(temp)

        pred = np.argmax(temp, axis=1)
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        result = {}

        i = 0
        for layer in filter(lambda x: isinstance(x, FullyConnectedLayer) or isinstance(x, ConvolutionalLayer), self.layers):
            param = layer.params()
            if param:
                keyw = 'W{}'.format(i)
                keyb = 'B{}'.format(i)
                result[keyw] = param['W']
                result[keyb] = param['B']
                i += 1
                
        return result
