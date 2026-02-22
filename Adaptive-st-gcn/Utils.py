import configparser
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from scipy.sparse.linalg import eigs
import os
import yaml
import configparser

# Read configuration file
"""
def ReadConfig(configfile):
    config = configparser.ConfigParser()
    print('Config: ', configfile)
    config.read(configfile)
    cfgPath = config['path']
    cfgTrain = config['train']
    cfgModel = config['model']
    return cfgPath, cfgTrain, cfgModel

"""


def ReadConfig(configfile):
    print('Config: ', configfile)

    ext = os.path.splitext(configfile)[1].lower()
    if ext in [".yaml", ".yml"]:
        with open(configfile, "r") as f:
            cfg = yaml.safe_load(f)

        # return 3 dicts just like before
        cfgPath = cfg.get("path", {})
        cfgTrain = cfg.get("train", {})
        cfgModel = cfg.get("model", {})
        return cfgPath, cfgTrain, cfgModel

    # fallback: INI/.config  the old one
    config = configparser.ConfigParser()
    config.read(configfile)
    cfgPath = config["path"]
    cfgTrain = config["train"]
    cfgModel = config["model"]
    return cfgPath, cfgTrain, cfgModel


# Add context to the origin data and label

def AddContext(x, context, label=False, dtype=float):
    ret = []
    assert context % 2 == 1, "context value error."

    cut = int(context / 2)
    if label:
        for p in range(len(x)):
            tData = x[p][cut:x[p].shape[0] - cut]
            ret.append(tData)
    else:
        for p in range(len(x)):
            tData = np.zeros([x[p].shape[0] - 2 * cut, context, x[p].shape[1], x[p].shape[2]], dtype=dtype)
            for i in range(cut, x[p].shape[0] - cut):
                tData[i - cut] = x[p][i - cut:i + cut + 1]
            ret.append(tData)
    return ret


# TIME-POINT LEVEL: this is for time point level prediction
def AddContextLabelSeq(y_list, context, dtype=np.int32, aggregation='center'):
    """
    Convert window labels into sequences aligned with AddContext(x, context).

    Supports TWO modes:
    1. WINDOW-LEVEL: y_list elements are (n_win, 1) or (n_win,)
    2. TIME-POINT LEVEL: y_list elements are (n_win, samples_per_window)

    For time-point labels, we aggregate using the specified method:
    - 'center': Use only the center sample of each window (default)
    - 'mean': Average all samples (gives probability)
    - 'max': Window is positive if ANY sample is positive

    Input:
      y_list: list of arrays
              Window-level: each shape (n_win, 1) or (n_win,)
              Time-point level: each shape (n_win, samples_per_window)
      context: odd int (e.g., 5)
      aggregation: 'center', 'mean', or 'max'

    Output:
      list of arrays, each shape (n_win - context + 1, context, 1)
      where y_seq[i] = y[i : i + context] after aggregation
    """
    assert context % 2 == 1, "context must be odd"
    ret = []

    for y in y_list:
        y = np.asarray(y)

        # Detect if this is time-point level or window level
        if y.ndim == 2 and y.shape[1] > 1:
            # TIME-POINT LEVEL: (n_win, samples_per_window)
            print(f"[INFO] Detected time-point labels: shape {y.shape}, using '{aggregation}' aggregation")

            if aggregation == 'center':
                # Extract center sample from each window
                center_idx = y.shape[1] // 2
                y_agg = y[:, center_idx:center_idx + 1]  # (n_win, 1)
            elif aggregation == 'mean':
                # Average across all samples (gives probability for binary labels)
                y_agg = y.mean(axis=1, keepdims=True).astype(np.float32)
            elif aggregation == 'max':
                # Window is positive if ANY sample is positive
                y_agg = y.max(axis=1, keepdims=True).astype(dtype)
            else:
                raise ValueError(f"Invalid aggregation: {aggregation}")
        else:
            # WINDOW-LEVEL: (n_win, 1) or (n_win,)
            print(f"[INFO] Detected window-level labels: shape {y.shape}")
            if y.ndim == 1:
                y_agg = y[:, None]  # (n_win, 1)
            else:
                y_agg = y  # already (n_win, 1)

        n = y_agg.shape[0]
        out_n = n - context + 1
        y_seq = np.zeros((out_n, context, 1), dtype=dtype)

        for i in range(out_n):
            y_seq[i, :, 0] = y_agg[i:i + context, 0]

        ret.append(y_seq)

    return ret


# Print score between Ytrue and Ypred
# savePath=None -> console, else to Result.txt

def PrintScore(true, pred, savePath=None, average='macro'):
    if savePath is None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'a+')

    F1 = metrics.f1_score(true, pred, average=None)  # length = number of classes

    # --- header and class names adapt to however many classes are present ---
    n_classes = len(F1)
    if n_classes == 2:
        class_names = ['NoSpindle', 'Spindle']
    else:
        class_names = [f'Class{i}' for i in range(n_classes)]

    # Main scores
    print("Main scores:")
    header = 'Acc\tF1S\tKappa\t' + '\t'.join([f'F1_{c}' for c in class_names])
    print(header, file=saveFile)

    values = [metrics.accuracy_score(true, pred),
              metrics.f1_score(true, pred, average=average),
              metrics.cohen_kappa_score(true, pred)] + list(F1)
    fmt = '\t'.join(['%.4f'] * len(values))
    print(fmt % tuple(values), file=saveFile)

    # Classification report
    print("\nClassification report:", file=saveFile)
    print(metrics.classification_report(true, pred, target_names=class_names), file=saveFile)

    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true, pred), file=saveFile)

    # Overall scores
    print('\n    Accuracy\t', metrics.accuracy_score(true, pred), file=saveFile)
    print(' Cohen Kappa\t', metrics.cohen_kappa_score(true, pred), file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=average), '\tAverage =', average, file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=average), '\tAverage =', average, file=saveFile)

    # Results of each class
    print('\nResults of each class:', file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=None), file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=None), file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=None), file=saveFile)

    if savePath is not None:
        saveFile.close()
    return


# Print confusion matrix and save

def ConfusionMatrix(y_true, y_pred, classes, savePath, title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Confusion matrix'
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n = cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Confusion matrix")
    print(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j] * 100, '.2f') + '%\n' + format(cm_n[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(savePath + title + ".png")
    plt.show()
    return ax


# Draw ACC / loss curve and save

def VariationCurve(fit, val, yLabel, savePath, figsize=(9, 6)):
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(fit) + 1), fit, label='Train')
    plt.plot(range(1, len(val) + 1), val, label='Val')
    plt.title('Model ' + yLabel)
    plt.xlabel('Epochs')
    plt.ylabel(yLabel)
    plt.legend()
    plt.savefig(savePath + 'Model_' + yLabel + '.png')
    plt.show()
    return


# compute \tilde{L}

def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    ----------
    Parameters
    W: np.ndarray, shape is (N, N), N is the num of vertices
    ----------
    Returns
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])


# compute a list of chebyshev polynomials from T_0 to T_{K-1}

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    ----------
    Parameters
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    ----------
    Returns
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    '''
    N = L_tilde.shape[0]
    cheb_polynomials = np.array([np.identity(N), L_tilde.copy()])
    for i in range(2, K):
        cheb_polynomials = np.append(cheb_polynomials,
                                     [2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2]], axis=0)
    return cheb_polynomials