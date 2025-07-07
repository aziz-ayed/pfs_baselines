import numpy as np
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

def harrell(times, events, risks):
    return concordance_index_censored(events.astype(bool), times, risks)[0]

def td_auc(t_train, e_train, t_val, e_val, r_val, eval_times):
    y_tr = Surv.from_arrays(e_train.astype(bool), t_train)
    y_va = Surv.from_arrays(e_val .astype(bool), t_val )
    auc, _ = cumulative_dynamic_auc(y_tr, y_va, r_val, eval_times)
    return auc.mean()