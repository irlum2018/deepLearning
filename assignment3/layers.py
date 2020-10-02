import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    
    return loss, grad

def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    
    assert predictions.ndim in [1, 2]

    if predictions.ndim == 1:
        exps = np.exp(predictions - np.max(predictions))
        return exps / np.sum(exps)
    else:
        exps = np.exp(predictions - np.max(predictions, axis=1).reshape(-1, 1))
        return exps / np.sum(exps, axis=1).reshape(-1, 1)

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    eps = 1e-9
    probs = np.clip(probs, eps, 1.0 - eps)

    if probs.ndim == 1:
        return -1 * np.log(probs[target_index])
    else:
        return -1 * np.sum(np.log(probs[np.arange(probs.shape[0]), target_index.flatten()])) / probs.shape[0]


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    dprediction = probs.copy()

    if len(predictions.shape) == 1:
        dprediction[target_index] -= 1
    else:
        dprediction[np.arange(predictions.shape[0]), target_index.flatten()] -= 1
        dprediction /= predictions.shape[0]

    return loss, dprediction

class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        result = np.maximum(0, X)
        return result

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = d_out * (self.X >= 0)
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        result = np.dot(X, self.W.value) + self.B.value
        
        return result

    def backward(self, d_out):
        # TODO copy from the previous assignment
        # out = x*w+b; db = (1,1,1, ...,1)* d_out
        
        self.B.grad += np.sum(d_out, axis=0).reshape(1, -1)
        self.W.grad += self.X.T.dot(d_out)
        d_input = np.dot(d_out, self.W.value.T)
        
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )
        #self.B = Param(np.zeros(out_channels))
        self.B = Param(0.001 * np.random.randn(1, out_channels))
        
        self.padding = padding
        self.stride = 1


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        fs = self.filter_size
        in_ch = self.in_channels
        out_ch = self.out_channels
        padd = self.padding 
        stride = self.stride
        out_height = int(np.floor((height - fs + 2*padd)/stride)) +1
        out_width = int(np.floor((width - fs + 2*padd)/stride)) +1
        
        #create padded copy of X frame
        pad_X=np.zeros((batch_size,X.shape[1]+2*padd,X.shape[2]+2*padd,channels))
        pad_X[:,padd:padd+X.shape[1],padd:padd+X.shape[2],:]= np.copy(X)
        
        #store X and pad_X
        self.X_cache = (X, pad_X)
        
        # setup variables that hold the result
        result = np.zeros((batch_size, out_height, out_width, out_ch))
        
        # and one x/y location at a time in the loop below
        for y in range(out_height):
            for x in range(out_width):
                # Implement forward pass for specific location
                
                recept_field = np.copy(pad_X[:,y:y+fs,x:x+fs,:])
                
                # convert input @ (x,y) into 'I' (batch_size, filter_size*filter_size*input_channels)
                recept_field=recept_field.reshape(batch_size,fs*fs*in_ch )
                # we'll have out_ch W[fs,fs, in_ch, out_ch], and B (out_ch,)
                fcl = FullyConnectedLayer(fs*fs*in_ch, out_ch)
                
                # convert weights into 'W' (filter_size*filter_size*input_channels, output_channels)
                # set  W and B for fully connected layer 
                fcl.W.value = self.W.value.reshape(fs*fs*in_ch, out_ch)
                # fc weights and bias
                fcl.B.value = self.B.value[0]
                
                #output
                temp = fcl.forward(recept_field)
                result[:,y,x,:] = np.copy(temp)

        return result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients
        fs = self.filter_size
        in_ch = self.in_channels
        out_ch = self.out_channels

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        
        (X, pad_X) = self.X_cache
        batch_size, height, width, _ = X.shape
        
        _, out_height, out_width, _ = d_out.shape
        d_in = np.zeros_like(pad_X)
        padd = self.padding 
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
  
                recept_field = np.copy(pad_X[:,y:y+fs,x:x+fs,:])
                # convert input  at x, y into 'I' (batch_size, filter_size*filter_size*input_channels) 
                recept_field=recept_field.reshape(batch_size,fs*fs*in_ch )
                # we'll have W(fs,fs,in_ch,out_ch), and B (out_ch,) - column vector 
                fcl = FullyConnectedLayer(fs*fs*in_ch, out_ch)
                fcl.W.value = self.W.value.reshape(fs*fs*in_ch, out_ch)
                # fc weights and bias
                fcl.B.value = self.B.value[0]
                fcl.X = recept_field
                
                # get the gradident
                temp = fcl.backward(np.copy(d_out[:,y,x,:]))
                # Aggregate gradients for the the parameters (W and B)
                
                grad = fcl.W.grad.reshape(fs,fs,in_ch,out_ch)
                #print("grad",grad.shape)
                
                # Aggregate gradients for the input 
                # propagate d_out only in x_slice position by element-wise multiplication
                self.W.grad += grad
                
                #print("fcl.B.grad.shape",fcl.B.grad.shape)
                self.B.grad += np.copy(fcl.B.grad.flatten())
                
                # Aggregate gradients for the input 
                d_in[:,y:y+fs,x:x+fs,:] += np.copy(temp.reshape((batch_size,fs,fs,in_ch)))
                
        #get rid of padding
        d_in_unpadded = d_in[:,padd:padd+height,padd:padd+width,:]
        return d_in_unpadded
        

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        
        ps = self.pool_size
        stride = self.stride
        out_height = int(np.floor((height - ps )/stride)) +1
        out_width = int(np.floor((width - ps )/stride)) +1

        #store X 
        self.X = X

        # setup variables that hold the result
        result = np.zeros((batch_size, out_height, out_width, channels))
        
        # and one x/y location at a time in the loop below
        for y in range(out_height):
            for x in range(out_width):
                # Implement forward pass for specific location
                
                recept_field = np.copy(X[:,y:y+ps,x:x+ps,:])
                max_slice = np.amax(recept_field,axis=(1, 2))
                result[:,y,x,:] = np.copy(max_slice)
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        
        ps = self.pool_size
        
        _, out_height, out_width, _ = d_out.shape
        d_in = np.zeros_like(self.X)
        
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                recept_field = np.copy(self.X[:,y:y+ps,x:x+ps,:])
                max_slice = np.amax(recept_field,axis=(1, 2))
                
                # filter out max value of receptive field with True
                mask = (recept_field == np.amax(recept_field, (1, 2))[:, np.newaxis, np.newaxis, :])
                # get the gradient and create new axis with y,x values
                grad = d_out[:,y,x,:][:, np.newaxis, np.newaxis, :]
                
                
                # Aggregate gradients for the input 
                # propagate d_out only in max_slice position by element-wise multiplication
                d_in[:,y:y+ps,x:x+ps,:] += mask*grad

                
        return d_in


    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        return X.reshape(batch_size,height*width*channels)

    def backward(self, d_out):
        
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
