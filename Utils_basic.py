import numpy as np

from Utils_embed import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc

  
def metrics(preds, labels):

    pred_labels = np.array([0 if i < 0.5 else 1 for i in preds])
    acc = round(accuracy_score(labels, pred_labels), 4)
    auroc = round(roc_auc_score(labels, preds), 4)
    (precisions, recalls, thresholds) = precision_recall_curve(labels, preds)
    auprc = round(auc(recalls, precisions), 4)
    return acc, auroc, auprc
        
