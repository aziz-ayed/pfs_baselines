import numpy as np
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

def harrell(times, events, risks):
    return concordance_index_censored(events.astype(bool), times, risks)[0]

def td_auc(t_train, e_train, t_val, e_val, r_val, eval_times):
    """
    Calculates time-dependent AUC.
    
    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of AUC scores for each time in eval_times.
            - float: The mean of the AUC scores.
    """
    y_tr = Surv.from_arrays(e_train.astype(bool), t_train)
    y_va = Surv.from_arrays(e_val.astype(bool), t_val)
    
    auc_scores, _ = cumulative_dynamic_auc(y_tr, y_va, r_val, eval_times)
    
    return auc_scores, auc_scores.mean()

