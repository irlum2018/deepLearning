import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0] # X test
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # TODO: Fill dists[i_test][i_train]
                dists[i_test,i_train]=np.sum(np.abs(X[i_test]-self.train_X[i_train]))
        return  dists     
                

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            dists[i_test]=np.sum(np.abs(X[i_test]-self.train_X), axis = 1)
        return  dists
         

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        # TODO: Implement computing all distances with no loops!
        # get another dimension
        X1 = X[:,np.newaxis]
        # subtraction will populate      
        dists = np.sum(np.abs( X1 - self.train_X), axis = 2)
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        classifier_knn =self.k
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            dist_sorted = sorted(dists[i])
            the_dist =dist_sorted[:classifier_knn]
            num_of_true=0
            num_of_false=0
            
            for val in the_dist:
                index = np.where(dists[i]==val)
                if len(index[0])>1:
                    index =[index[0][0]]
                 
                if  self.train_y[index]:
                    num_of_true+=1
                else:
                    num_of_false+=1   
                              
            min_label= np.where(dists[i] == np.amin(dists[i]))
            if num_of_true > num_of_false:
                pred[i]= True
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        classifier_knn =self.k
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            dist_sorted = sorted(dists[i])
            the_dist =dist_sorted[:classifier_knn]
            
            label_to_count = {} 
            for val in the_dist:
                index = np.where(dists[i]==val)
                if len(index[0])>1:
                    index =[index[0][0]]
                label=self.train_y[index][0][0]
                if label in label_to_count:
                    label_to_count[label]+=1
                else:
                    label_to_count[label]=1
            sorted_label_to_count = sorted(label_to_count.items(), key=lambda kv: kv[1], reverse = True)
            pred[i]= [key for key, value in sorted_label_to_count][0]
 
        return pred
