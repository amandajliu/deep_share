# Written by Yongxi Lu

""" Utility functions used to compute error """

import numpy as np

def compute_mle(scores, targets):
    """ Compute multi-label error """
    num_classes = targets.shape[1]
    err = np.empty((num_classes,), dtype=np.float32)
    err[:] = np.nan
    for c in xrange(num_classes):
        # negative label is reserved for "unknown", evaluation of those entries are skipped. 
        valid_ind = np.where(targets[:,c]>=0.0)[0]
        if len(valid_ind>0):
            err[c] = np.mean(((scores[valid_ind, [c]]>=0.5) != (targets[valid_ind, [c]]>=0.5)), axis=0)
    return err

def compute_neg_mle(scores, targets, false_pos=True):

    """ Compute multi-label error """
    num_classes = targets.shape[1]
    err = np.empty((num_classes,), dtype=np.float32)
    err[:] = np.nan
    for c in xrange(num_classes):

        # negative label is reserved for "unknown", evaluation of those entries are skipped. 
        valid_ind = np.where(targets[:,c]>=0.0)[0]
        pred_true = np.where(scores[valid_ind, [c]]>0.5, [True], [False])
        actual_true = np.where(targets[valid_ind, [c]]<=0.5, [True], [False])
        if not false_pos:
            err[c] = np.sum(np.logical_and(pred_true==False,actual_true==True))
        else:
            err[c] = np.sum(np.logical_and(pred_true==True, actual_true==False))
        # if len(valid_ind>0):
        #     err[c] = np.mean(((scores[valid_ind, [c]]<=0.5) != (targets[valid_ind, [c]]<=0.5)), axis=0)
    return err
