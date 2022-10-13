from sklearn import metrics


def calc_acc(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)


def calc_f1(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred, zero_division=True), \
        metrics.recall_score(y_true, y_pred, zero_division=True), \
            metrics.f1_score(y_true, y_pred, zero_division=True)


def calc_auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)


def calc_ks(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    return max(tpr - fpr)


def precision_at_r9(y_true, y_pred):
    import bisect
    from sklearn.metrics import precision_recall_curve
    p, r, tresholds = precision_recall_curve(y_true, y_pred)
    ind = bisect.bisect(r, 0.9)
    if ind >= len(p)-1:
        return r[-2], p[-2], tresholds[-1]
    else:
        return r[ind], p[ind], tresholds[ind]


def recall_at_p9(y_true, y_pred):
    import bisect
    from sklearn.metrics import precision_recall_curve
    p, r, tresholds = precision_recall_curve(y_true, y_pred)
    ind = bisect.bisect(p, 0.9)
    if ind >= len(p)-1:
        return r[-2], p[-2], tresholds[-1]
    else:
        return r[ind], p[ind], tresholds[ind]