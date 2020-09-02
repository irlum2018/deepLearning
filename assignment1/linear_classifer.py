import numpy as np


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
    probs = predictions.copy()
    if len(predictions.shape) == 1:
        probs -= np.max(predictions)
        probs = np.exp(probs) / np.sum(np.exp(probs))
    else:
        probs -= np.max(predictions, axis=1).reshape(-1, 1)
        probs = np.exp(probs) / np.sum(np.exp(probs), axis=1).reshape(-1, 1)

    return probs

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
    if len(probs.shape) == 1:
        loss = - np.log(probs[target_index])
    else:
        loss = -np.sum(np.log(probs[np.arange(probs.shape[0]), target_index.flatten()]))
    return loss


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
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    if len(predictions.shape) == 1:
        # target_index -- ненулевой индекс, поэтому 1 вычитается только в нём
        probs[target_index] -= 1
    else:
        probs[np.arange(probs.shape[0]), target_index.flatten()] -= 1

    dprediction = probs
    return loss, dprediction
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
    loss = reg_strength*np.sum(np.square(W))
    grad = 2*reg_strength*W

    return loss, grad

def testZWgrad(X, W):
     prs = np.dot(X, W)
     loss = np.trace(prs)
     #print("DOT PRODUCT",prs, prs.shape)
     X_T = X.transpose()
     DW = X_T
     return loss, DW  

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of loss by weight
    '''
    predictions = np.dot(X, W)
    loss, DZ = softmax_with_cross_entropy(predictions,target_index)
    X_T = np.transpose(X)
    DW= np.dot(X_T,DZ)
    
    return loss, DW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            loss=0.0
            for fold in range(len(sections)):
                test_x=X[batches_indices[fold]]
                test_y=y[batches_indices[fold]]
                loss_lin, dW_lin = linear_softmax(test_x, self.W, test_y)
                loss_r, dW_r = l2_regularization(self.W, reg)
                loss_tot =loss_lin+loss_r
                dW_tot = dW_lin+dW_r
                self.W = self.W-learning_rate*dW_tot
                loss_history.append(loss_tot)
            #print("Epoch %i, loss: %f" % (epoch, loss_tot))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        scores = np.dot(X, self.W )
        pred = softmax(scores)
        y_pred=np.argmax(pred, axis=1)
        return y_pred



                
                                                          

            

                
