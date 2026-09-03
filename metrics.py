"""Evaluation metrics for cue detection and scope resolution."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def f1_cues(y_true, y_pred):
    '''Needs flattened cues'''
    tp = sum([1 for i,j in zip(y_true, y_pred) if (i==j and i!=3)])
    fp = sum([1 for i,j in zip(y_true, y_pred) if (j!=3 and i==3)])
    fn = sum([1 for i,j in zip(y_true, y_pred) if (i!=3 and j==3)])
    if tp==0:
        prec = 0.0001
        rec = 0.0001
    else:
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {2*prec*rec/(prec+rec)}")
    return prec, rec, 2*prec*rec/(prec+rec)
    
    
def f1_scope(y_true, y_pred, level = 'token', average = 'macro'):
    '''Scope-resolution F1. Both arguments are lists of per-sentence label lists.

    Scope resolution is evaluated against gold cue annotations, so a prediction
    is only ever made where a cue exists and the scope-level precision is 1 by
    construction.

    level='token' -- F1 over the flattened per-token labels. `average` selects
    the sklearn averaging scheme: 'macro' (the "Macro F1 Average (Token-level)"
    of the paper's Section 4, averaging IN_SCOPE and OUT_OF_SCOPE equally),
    'binary' (the IN_SCOPE class alone, which is what the tables' generating
    code actually reports), or any other sklearn value.
    level='scope' -- exact-match over whole scopes: a sentence counts as a true
    positive only if every one of its tokens is labelled correctly.

    Returns the F1 for level='token', and (precision, recall, F1) for
    level='scope'.
    '''
    if level == 'token':
        # These were previously flattened with the two `for` clauses in the
        # wrong order ([i for i in j for j in y_true]), which reads `j` before
        # it is bound and so raised NameError -- the token-level branch never
        # ran. The comprehension binds the outer loop first.
        true_flat = [i for sent in y_true for i in sent]
        pred_flat = [i for sent in y_pred for i in sent]
        f1 = f1_score(true_flat, pred_flat, average=average)
        print(f"F1 Score (token-level, {average}): {f1}")
        return f1
    elif level == 'scope':
        tp = 0
        fn = 0
        fp = 0
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == y_p:
                tp+=1
            else:
                fn+=1
        prec = 1
        rec = tp/(tp+fn)
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1 Score: {f1}")
        return prec, rec, f1
    else:
        raise ValueError("level must be one of 'token' and 'scope'")

def report_per_class_accuracy(y_true, y_pred):
    labels = list(np.unique(y_true))
    lab = list(np.unique(y_pred))
    labels = list(np.unique(labels+lab))
    n_labels = len(labels)
    data = pd.DataFrame(columns = labels, index = labels, data = np.zeros((n_labels, n_labels)))
    for i,j in zip(y_true, y_pred):
        data.at[i,j]+=1
    print(data)
    
def flat_accuracy(preds, labels, input_mask = None):
    pred_flat = [i for j in preds for i in j]
    labels_flat = [i for j in labels for i in j]
    return sum([1 if i==j else 0 for i,j in zip(pred_flat,labels_flat)]) / len(labels_flat)
    

def flat_accuracy_positive_cues(preds, labels, input_mask = None):
    pred_flat = [i for i,j in zip([i for j in preds for i in j],[i for j in labels for i in j]) if (j!=4 and j!=3)]
    labels_flat = [i for i in [i for j in labels for i in j] if (i!=4 and i!=3)]
    if len(labels_flat) != 0:
        return sum([1 if i==j else 0 for i,j in zip(pred_flat,labels_flat)]) / len(labels_flat)
    else:
        return None

def scope_accuracy(preds, labels):
    correct_count = 0
    count = 0
    for i,j in zip(preds, labels):
        if i==j:
            correct_count+=1
        count+=1
    return correct_count/count
