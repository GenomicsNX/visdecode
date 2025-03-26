import numpy as np
import Levenshtein

VAR_TYPES_CLASSES = ["quantitative", "temporal", "nominal", "ordinal"]
MARK_TYPE_CLASSES = ["hbar", "vbar", "line", "point", "pie"]

def dict_mean(dict):

    '''computes the mean across all values in the dictionary'''

    dict_values = [v for v in dict.values() if v is not None]
    return (sum(dict_values) / len(dict_values)) if len(dict_values) > 0 else 0

def multiclass_confusion_matrix(samples, classes):
    
    ''' creates confusion matrix for multiclass classification \n
        "samples" : list of tuples such as: [(pred, gt), (pred, gt), ...] \n
        "classes" : list of all possible classes 
    '''

    n = len(classes)
    confusion_mat = np.zeros((n,n))

    for pred_class, gt_class in samples:
        if pred_class in classes:

            i = classes.index(gt_class)
            j = classes.index(pred_class)

            confusion_mat[i,j] += 1

    return confusion_mat

def f1_score(multiclass_confusion_mat, classes, average = False):
    
    ''' computes f1-score from a confusion matrix \n
        "classes" : list of all possible classes \n'''

    n = multiclass_confusion_mat.shape[0]
    scores = {}

    for i in range(len(classes)):
        
        TP, FN, FP, = 0.0, 0.0, 0.0

        for col in range(n):

            if i == col: 
                TP += multiclass_confusion_mat[i, col]
            else:
                FN += multiclass_confusion_mat[i, col]
                    
        for row in range(n):
            if i != row: FP += multiclass_confusion_mat[row, i]

        f1 = None

        if TP + FN + FP > 0:
            precision   = np.round(TP / (TP + FP), 2) if TP + FP > 0 else 0 
            recall      = np.round(TP / (TP + FN), 2) if TP + FN > 0 else 0

            f1 = np.round(2 * (precision * recall) / (precision + recall), 2) if precision + recall > 0 else 0

        scores[classes[i]] = f1

    if average: return dict_mean(scores)
    return scores

def nld_score(samples):
    
    '''computes normalized levenshtein distance average for all samples \n
        "samples" : list of tuples such as: [(pred, gt), (pred, gt), ...] \n'''

    scores = [Levenshtein.ratio(sample[0], sample[1]) for sample in samples]
    mean_score = np.round(np.mean(scores), 4)

    return mean_score