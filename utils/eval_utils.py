import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_accuracy_anomaly(output, target):
    """
     Calculates the classification accuracy.
    :param target: Tensor of correct labels of size [batch_size]
    :param output: Tensor of model predictions of size [batch_size, num_classes]
    :return: prediction accuracy
    """
    num_samples = target.size(0)
    num_correct = torch.sum(torch.argmax(target, dim=1) == torch.argmax(output, dim=1))
    accuracy = num_correct.float() / num_samples
    return accuracy.detach().cpu().numpy()



def binary_accuracy(output, target):
    """
     Calculates the classification accuracy.
    :param target: Tensor of correct labels of size [batch_size]
    :param output: Tensor of model predictions of size [batch_size, num_classes]
    :return: prediction accuracy
    """
    num_samples = target.size(0)
    score = torch.sigmoid(output)
    preds = torch.argmax(output, dim=1)# score > 0.5
    num_correct = torch.sum(target.squeeze() == preds)
    accuracy = num_correct.float() / num_samples
    return accuracy.detach().cpu().numpy()


def compute_metrics_anomaly(output, target):
    precision = dict()
    recall = dict()
    average_precision = dict()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision["micro"], recall["micro"], _ = precision_recall_curve(target.cpu().detach().numpy().ravel(),
                                                                    output.cpu().detach().numpy().ravel())
    average_precision["micro"] = average_precision_score(target.cpu().detach().numpy(),
                                                         output.cpu().detach().numpy(),
                                                         average="micro")
    aupr = auc(recall["micro"], precision["micro"])

    fpr["micro"], tpr["micro"], _ = roc_curve(target.cpu().detach().numpy().ravel(),
                                              output.cpu().detach().numpy().ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return aupr, roc_auc["micro"]

def compute_metrics(outputs, targets):
    metrics = roc_auc_score(targets.detach().cpu().numpy(), outputs.detach().cpu().numpy())
    return metrics


def plot_roc_curve(binarized_y_true, y_proba_pred, class_name):
    fpr, tpr, thresholds = roc_curve(binarized_y_true, y_proba_pred)
    auc = roc_auc_score(binarized_y_true, y_proba_pred)
    auc = round(auc, 2)

    plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc})")

    return fpr, tpr, auc


def compute_accuracy_multitask(output, target):
    """
     Calculates the classification accuracy.
    :param target: Tensor of correct labels of size [batch_size, numClasses]
    :param output: Predicted scores (logits) by the model.
            It should have the same dimensions as target
    :return: accuracy: average accuracy over the samples of the current batch for each condition
    """
    num_samples = target.size(0)
    correct_pred = target.eq(output.round().long())
    accuracy = torch.sum(correct_pred, dim=0)
    return accuracy.cpu().numpy() * (100. / num_samples)
