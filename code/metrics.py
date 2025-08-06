# roc working???  i have some doubts




"""
this file  Implement all metric needed """

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from fontTools.misc.cython import returns


class CustomMetrics:

    def __init__(self):

        self.reset()  # initializing

    def reset(self):
        # Reseting all stored predictions (0 /1) and targets (true label from json 0/1)"""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []

    def add_batch(self, predictions, targets, probabilities=None):

        # Convert tensors to numpy if needed
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        if probabilities is not None and torch.is_tensor(probabilities):
            probabilities = probabilities.cpu().numpy()

        # Flatten arrays to ensure 1D
        predictions = predictions.flatten()
        targets = targets.flatten()

        # Store the batch data
        self.all_predictions.extend(predictions)
        self.all_targets.extend(targets)

        if probabilities is not None:
            probabilities = probabilities.flatten()
            self.all_probabilities.extend(probabilities)

    def calculate_confusion_matrix(self, y_true, y_pred):
        #confusion matrix :


        # Convert to numpy arrays
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)

        # Calculate each component of confusion matrix
        # True Positives: predicted 1, actual 1
        tp = np.sum((y_pred == 1) & (y_true == 1))

        # True Negatives: predicted 0, actual 0
        tn = np.sum((y_pred == 0) & (y_true == 0))

        # False Positives: predicted 1, actual 0
        fp = np.sum((y_pred == 1) & (y_true == 0))

        # False Negatives: predicted 0, actual 1
        fn = np.sum((y_pred == 0) & (y_true == 1))

        return tn, fp, fn, tp

    def calculate_accuracy(self, tn, fp, fn, tp):
        #accuracy - How many predictions were correct out of total predictions

        total = tn + fp + fn + tp
        if total == 0:
            return 0.0
        return (tp + tn) / total

    def calculate_precision(self, tn, fp, fn, tp):
        # precision - Of all spindles we detected, how many were real spindles?

        denominator = tp + fp
        if denominator == 0:
            return 0.0
        return tp / denominator

    def calculate_recall(self, tn, fp, fn, tp):
        # recall (sensitivity) = TP / (TP + FN)  - Of all real spindles, how many did we detect?

        denominator = tp + fn
        if denominator == 0:
            return 0.0
        return tp / denominator

    def calculate_specificity(self, tn, fp, fn, tp):

       # Calculate specificity = TN / (TN + FP) - Of all non-spindle periods, how many did we correctly identify?

        denominator = tn + fp
        if denominator == 0:
            return 0.0
        return tn / denominator

    def calculate_f1_score(self, precision, recall):
        # F1 score = 2 * (precision * recall) / (precision + recall) - F1 score is harmonic mean of precision and recall

        denominator = precision + recall
        if denominator == 0:
            return 0.0
        return 2 * (precision * recall) / denominator

    def calculate_roc_curve(self, y_true, y_proba):
        ''' ROC curve :

            #fpr: False positive rates
            #tpr: True positive rates
            #thresholds: Threshold values
        # Convert to numpy arrays'''
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)

        # Get unique thresholds (sorted in descending order)
        thresholds = np.unique(y_proba)
        thresholds = np.sort(thresholds)[::-1]

        # Add extreme thresholds
        thresholds = np.concatenate([[1.0], thresholds, [0.0]])

        # Calculate TPR and FPR for each threshold
        tpr_list = []
        fpr_list = []

        for threshold in thresholds:
            # Make predictions with this threshold
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate confusion matrix
            tn, fp, fn, tp = self.calculate_confusion_matrix(y_true, y_pred)
            # errror?????
            # Calculate TPR (recall) and FPR
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        return np.array(fpr_list), np.array(tpr_list), thresholds

    def calculate_auc_roc(self, fpr, tpr):
        """
        Calculate Area Under ROC Curve using trapezoidal rule"""
        # Sort by FPR to ensure proper integration
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]

        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(fpr_sorted)):
            # Width of trapezoid
            width = fpr_sorted[i] - fpr_sorted[i - 1]
            # Average height of trapezoid
            height = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
            # Add area of this trapezoid
            auc += width * height

        return auc # if we get result 1.0 Perfect classifier  , 0.5 - Random classifier

    def calculate_precision_recall_curve(self, y_true, y_proba):

       # Precision-Recall curve :

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)

        # Get unique thresholds (sorted in descending order)
        thresholds = np.unique(y_proba)
        thresholds = np.sort(thresholds)[::-1]

        # Calculate precision and recall for each threshold
        precision_list = []
        recall_list = []

        for threshold in thresholds:
            # Make predictions with this threshold
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate confusion matrix
            tn, fp, fn, tp = self.calculate_confusion_matrix(y_true, y_pred)

            # Calculate precision and recall
            precision = self.calculate_precision(tn, fp, fn, tp)
            recall = self.calculate_recall(tn, fp, fn, tp)

            precision_list.append(precision)
            recall_list.append(recall)

        return np.array(precision_list), np.array(recall_list), thresholds

    def calculate_auc_pr(self, precision, recall):
        """
        Calculate Area Under Precision-Recall Curve- i thought AUC-PR is better than AUC-ROC for imbalanced datasets
        - this will Focuses on positive class performance

        """
        # Sort by recall (ascending order)
        sorted_indices = np.argsort(recall)
        recall_sorted = recall[sorted_indices]
        precision_sorted = precision[sorted_indices]

        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(recall_sorted)):
            # Width of trapezoid
            width = recall_sorted[i] - recall_sorted[i - 1]
            # Average height of trapezoid
            height = (precision_sorted[i] + precision_sorted[i - 1]) / 2
            # Add area of this trapezoid
            auc += width * height

        return auc

    def calculate_all_metrics(self):

        #Calculate all metrics from stored predictions and targets

        if len(self.all_predictions) == 0:
            raise ValueError("No predictions added. Use add_batch() first.")

        # Convert to numpy arrays
        y_true = np.array(self.all_targets, dtype=int)
        y_pred = np.array(self.all_predictions, dtype=int)

        # Calculate confusion matrix components
        tn, fp, fn, tp = self.calculate_confusion_matrix(y_true, y_pred)

        # Calculate basic metrics
        accuracy = self.calculate_accuracy(tn, fp, fn, tp)
        precision = self.calculate_precision(tn, fp, fn, tp)
        recall = self.calculate_recall(tn, fp, fn, tp)
        specificity = self.calculate_specificity(tn, fp, fn, tp)
        f1_score = self.calculate_f1_score(precision, recall)

        # Create results dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
        }
        # ROC & AUC
        if self.all_probabilities:
            fpr, tpr, _ = self.calculate_roc_curve(y_true, np.array(self.all_probabilities))
            auc_roc = self.calculate_auc_roc(fpr, tpr)
            metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
            metrics['auc_roc'] = auc_roc

            # PR & AUC
            precision_curve, recall_curve, _ = self.calculate_precision_recall_curve(y_true,
                                                                                     np.array(self.all_probabilities))
            auc_pr = self.calculate_auc_pr(precision_curve, recall_curve)
            metrics['pr_curve'] = {'precision': precision_curve, 'recall': recall_curve}
            metrics['auc_pr'] = auc_pr

        return metrics

    def find_optimal_threshold(self):

        #trying to find  optimal threshold that maximizes F1 score


        if len(self.all_probabilities) == 0:
            return 0.5  # Default threshold if no probabilities stored

        y_true = np.array(self.all_targets, dtype=int)
        y_proba = np.array(self.all_probabilities)

        best_threshold = 0.5
        best_f1 = 0.0

        # Test thresholds from 0.05 to 0.95 in steps of 0.05
        for threshold in np.arange(0.05, 0.95, 0.05):
            # Convert probabilities to binary predictions using this threshold
            y_pred = (y_proba > threshold).astype(int)

            # Calculate confusion matrix for this threshold
            tn, fp, fn, tp = self.calculate_confusion_matrix(y_true, y_pred)

            # Calculate F1 score for this threshold
            precision = self.calculate_precision(tn, fp, fn, tp)
            recall = self.calculate_recall(tn, fp, fn, tp)
            f1 = self.calculate_f1_score(precision, recall)

            # Update best threshold if this F1 is better
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def print_detailed_report(self, metrics, dataset_name="Dataset"):
        # detailed evaluation report
        print(f"\n{'=' * 50}")
        print(f"{dataset_name} Evaluation Report")
        print(f"{'=' * 50}")

        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1-Score:    {metrics['f1_score']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")

        if 'auc_roc' in metrics:
            print(f"AUC-ROC:     {metrics['auc_roc']:.4f}")
            print(f"AUC-PR:      {metrics['auc_pr']:.4f}")

        print(f"\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"[[{cm[0][0]:4d}, {cm[0][1]:4d}]")
        print(f" [{cm[1][0]:4d}, {cm[1][1]:4d}]]")
        print("  TN   FP")
        print("  FN   TP")

        print(f"\nDetailed Counts:")
        print(f"True Positives:  {metrics['true_positives']}")
        print(f"True Negatives:  {metrics['true_negatives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")

    def plot_confusion_matrix(self, metrics, save_path=None, title="Confusion Matrix"):

        #Ploting confusion matrix as heatmap

        cm = np.array(metrics['confusion_matrix'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Spindle', 'Spindle'],
                    yticklabels=['No Spindle', 'Spindle'])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_roc_curve(self, metrics, save_path=None, title="ROC Curve"):
        #Plot ROC curve

        if 'roc_curve' not in metrics:
            print("ROC curve data not available. Need probabilities for ROC curve.")
            return

        fpr = metrics['roc_curve']['fpr']
        tpr = metrics['roc_curve']['tpr']
        auc_roc = metrics['auc_roc']

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_roc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        plt.show()

    def plot_precision_recall_curve(self, metrics, save_path=None, title="Precision-Recall Curve"):

        #Plot Precision-Recall curve


        if 'pr_curve' not in metrics:
            print("PR curve data not available. Need probabilities for PR curve.....")
            return

        precision = metrics['pr_curve']['precision']
        recall = metrics['pr_curve']['recall']
        auc_pr = metrics['auc_pr']

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AUC = {auc_pr:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")

        plt.show()

    def plot_all_curves(self, metrics, save_dir=None, prefix=""):

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Plot confusion matrix
        cm_path = save_dir / f"{prefix}confusion_matrix.png" if save_dir else None
        self.plot_confusion_matrix(metrics, cm_path)

        # Plot ROC curve
        if 'roc_curve' in metrics:
            roc_path = save_dir / f"{prefix}roc_curve.png" if save_dir else None
            self.plot_roc_curve(metrics, roc_path)

        # Plot PR curv
        if 'pr_curve' in metrics:
            pr_path = save_dir / f"{prefix}pr_curve.png" if save_dir else None
            self.plot_precision_recall_curve(metrics, pr_path)

    def evaluate_model_with_threshold(self, model, data_loader, device, threshold=0.5):
        """
        Evaluate model on a dataset with specific threshold """

        model.eval()  # Set model to evaluation mode
        self.reset()  # Clear previous results

        with torch.no_grad():  # Disable gradient calculation for efficiency
            for data, targets in data_loader:
                # Move data to device (CPU or GPU)
                data, targets = data.to(device), targets.to(device)

                # Get model predictions (logits)
                outputs = model(data)

                # Convert logits to probabilities using sigmoid
                probabilities = torch.sigmoid(outputs)

                # Convert probabilities to binary predictions using threshold
                predictions = (probabilities > threshold).float()

                # Add this batch to our metrics calculator
                self.add_batch(predictions, targets, probabilities)

        # Calculate and return all metrics
        return self.calculate_all_metrics()