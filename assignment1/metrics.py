def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    num_samples=len(ground_truth)
    count =0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    ap = 0
    an = 0
    recall=0
    precision=0
    accuracy = 0
    f1=0
    for i in range(num_samples):
    	if ground_truth[i] ==True:
    	    ap+=1
    	    # correctly identified as true
    	    if prediction[i] ==True:
    	        tp+=1
    	    # incorrectly identified as negative
    	    else:
    	        fn+=1
    	else:
    	    an+=1
    	    # incorrectly identified as positive
    	    if prediction[i] ==True:
    	        fp+=1
    	    # correctly identified as negative
    	    else:
    	        tn+=1
    
    if tp!=0:
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        f1= 2*precision*recall/(recall+precision)
    '''
    else:
        print("0000",prediction,ground_truth)
    '''
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    num_samples=len(ground_truth)
    count =0
    accuracy =0
    for i in range(num_samples):
        if prediction[i]==ground_truth[i]:
            count+=1
    if count!=0:
        accuracy = count/num_samples
    return accuracy
