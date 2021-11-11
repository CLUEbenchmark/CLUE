import csv
import sys
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "csl":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "cmnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "ocnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "iflytek":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wsc":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "tnews":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "afqmc":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "copa":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
