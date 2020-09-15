import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        layers = []
        layers.append(FullyConnectedLayer(n_input, hidden_layer_size))
        layers.append(ReLULayer())
        layers.append(FullyConnectedLayer(hidden_layer_size, n_output))
        self.layers = layers

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for _, param in self.params().items():
            param.grad = np.zeros_like(param.value)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # forward pass
        temp = X
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
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        temp = X
        for layer in self.layers:
            temp = layer.forward(temp)

        pred = np.argmax(temp, axis=1)
        return pred

    def params(self):
        result = {}

        i = 0
        for layer in filter(lambda x: isinstance(x, FullyConnectedLayer), self.layers):
            param = layer.params()
            if param:
                result['W{}'.format(i)] = param['W']
                result['B{}'.format(i)] = param['B']
                i += 1
        return result
