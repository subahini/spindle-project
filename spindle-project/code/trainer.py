"""
Enhanced trainer with time-based evaluation and wandb integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from models import FocalLoss
from metrics import TimeBasedMetrics
from wandb_integration import ExperimentTracker

class EarlyStopping:
    """Enhanced early stopping with multiple metrics support"""

    def __init__(self, patience=10, min_delta=0.001, mode='max', monitor='f1_score'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # 'max' for f1_score, 'min' for loss
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False
        if self.mode == 'max':
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)
        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
        self.early_stop = self.counter >= self.patience
        return self.early_stop

class SpindleTrainer:
    """Enhanced trainer for time-based spindle detection"""

    def __init__(self, config, spindle_annotations: List[Tuple[float, float]]):
        self.config = config
        self.spindle_annotations = spindle_annotations
        self.time_metrics = TimeBasedMetrics(config)
        # Initialize experiment tracker
        if config.get('wandb.enabled', True):
            self.experiment_tracker = ExperimentTracker(config)
        else:
            self.experiment_tracker = None
        # Setup loss function
        self.criterion = self._setup_loss_function()
        print(f"Trainer initialized with {config.get('training.loss_function', 'bce')} loss")

    def _setup_loss_function(self):
        """Setup loss function based on configuration"""
        loss_type = self.config.get('training.loss_function', 'bce')
        if loss_type == 'focal':
            alpha = self.config.get('training.focal_alpha', 0.25)
            gamma = self.config.get('training.focal_gamma', 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == 'weighted_bce':
            # Will set pos_weight during training
            return nn.BCEWithLogitsLoss()
        elif loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

    def calculate_class_weights(self, train_loader):
        """Calculate class weights for imbalanced data"""
        print("Calculating class weights...")
        train_labels = []
        for _, labels in train_loader:
            train_labels.extend(labels.numpy().flatten())
        train_labels = np.array(train_labels)
        n_pos = int(np.sum(train_labels))
        n_neg = int(len(train_labels) - n_pos)
        if n_pos == 0:
            pos_weight_value = 1.0
        else:
            pos_weight_value = n_neg / n_pos
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(self.config.DEVICE)
        print(f"Class distribution - Positive: {n_pos}, Negative: {n_neg}")
        print(f"Positive class weight: {pos_weight_value:.3f}")
        # Log to wandb
        if self.experiment_tracker:
            self.experiment_tracker.logger.log_metrics({
                'data/positive_samples': n_pos,
                'data/negative_samples': n_neg,
                'data/positive_weight': pos_weight_value,
                'data/class_imbalance_ratio': n_neg / max(n_pos, 1)
            })
        return pos_weight

    def _setup_optimizer_and_scheduler(self, model):
        """Setup optimizer and learning rate scheduler"""
        optimizer_type = self.config.get('training.optimizer', 'adam')
        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        scheduler_type = self.config.get('training.scheduler', 'plateau')
        if scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.EPOCHS)
        else:
            scheduler = None
        return optimizer, scheduler

    def train_model(self, model, train_loader, val_loader, model_name="CNN"):
        """Train model with comprehensive time-based evaluation"""
        print(f"\n{'=' * 60}")
        print(f"Starting {model_name} Training with Time-Based Evaluation")
        print(f"{'=' * 60}")
        # Start experiment tracking
        if self.experiment_tracker:
            self.experiment_tracker.start_experiment(model)
        # Setup loss function with weights if needed
        if self.config.get('training.loss_function') == 'weighted_bce':
            pos_weight = self.calculate_class_weights(train_loader)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            if hasattr(self, 'calculate_class_weights'):
                self.calculate_class_weights(train_loader)  # Just for logging
        # Setup optimizer and scheduler
        optimizer, scheduler = self._setup_optimizer_and_scheduler(model)
        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=self.config.PATIENCE,
            mode=self.config.get('advanced.early_stopping_mode', 'max'),
            monitor=self.config.get('advanced.checkpoint_monitor', 'val_f1_score')
        )
        # Training history
        training_history = {
            'train_losses': [],
            'val_metrics_history': [],
            'learning_rates': [],
            'best_val_f1': 0.0,
            'best_model_state': None,
            'best_epoch': 0
        }
        start_time = time.time()
        grad_clip_norm = self.config.get('training.grad_clip_norm', 0.5)
        for epoch in range(self.config.EPOCHS):
            epoch_start = time.time()
            # === TRAINING PHASE ===
            model.train()
            epoch_losses = []
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.config.DEVICE), targets.to(self.config.DEVICE)
                optimizer.zero_grad()
                outputs = model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                # Gradient clipping
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
                epoch_losses.append(loss.item())
                # Log batch metrics periodically
                if (batch_idx + 1) % self.config.get('logging.log_every_n_steps', 100) == 0:
                    if self.experiment_tracker:
                        step = epoch * len(train_loader) + batch_idx
                        self.experiment_tracker.logger.log_training_step(epoch, step, loss.item())
            avg_train_loss = np.mean(epoch_losses)
            training_history['train_losses'].append(avg_train_loss)
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            # === VALIDATION PHASE ===
            val_metrics = self._evaluate_time_based(
                model, val_loader,
                start_time=self.config.VAL_START,
                end_time=self.config.VAL_END,
                threshold=self.config.get('evaluation.default_threshold', 0.5)
            )
            training_history['val_metrics_history'].append(val_metrics)
            # Track best model
            current_f1 = val_metrics.get('time_metrics', {}).get('f1_score', 0)
            if current_f1 > training_history['best_val_f1']:
                training_history['best_val_f1'] = current_f1
                training_history['best_model_state'] = model.state_dict().copy()
                training_history['best_epoch'] = epoch
            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_f1)
                else:
                    scheduler.step()
            epoch_time = time.time() - epoch_start
            # === LOGGING ===
            self._log_epoch_results(epoch, avg_train_loss, val_metrics, epoch_time, optimizer.param_groups[0]['lr'])
            # Log to experiment tracker
            if self.experiment_tracker:
                self.experiment_tracker.log_epoch(epoch, avg_train_loss, val_metrics.get('time_metrics', {}))
            # Early stopping check
            if early_stopping(current_f1):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best validation F1: {training_history['best_val_f1']:.4f} at epoch {training_history['best_epoch'] + 1}")
                break
            # Save checkpoint periodically
            if (epoch + 1) % self.config.get('logging.save_model_every_n_epochs', 10) == 0:
                self._save_checkpoint(model, optimizer, epoch, training_history, model_name)
        # Restore best model
        if training_history['best_model_state'] is not None:
            model.load_state_dict(training_history['best_model_state'])
            print(f"\nRestored best model from epoch {training_history['best_epoch'] + 1}")
        training_history['total_time'] = time.time() - start_time
        return {
            'model': model,
            'training_history': training_history,
            'final_metrics': val_metrics
        }

    def _evaluate_time_based(self, model, data_loader, start_time, end_time, threshold=0.5):
        """Evaluate model using time-based metrics"""
        try:
            results = self.time_metrics.evaluate_model_timeline(
                model=model,
                data_loader=data_loader,
                device=self.config.DEVICE,
                spindle_annotations=self.spindle_annotations,
                start_time=start_time,
                end_time=end_time,
                threshold=threshold
            )
            return results
        except Exception as e:
            print(f"Warning: Time-based evaluation failed: {e}")
            return {
                'time_metrics': {'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0},
                'segment_metrics': {'segment_f1': 0.0}
            }

    def _log_epoch_results(self, epoch, train_loss, val_metrics, epoch_time, learning_rate):
        """Log epoch results to console"""
        print(f"\nEpoch {epoch + 1}/{self.config.EPOCHS} ({epoch_time:.1f}s, lr={learning_rate:.2e})")
        print(f"  Train Loss: {train_loss:.4f}")
        if 'time_metrics' in val_metrics:
            tm = val_metrics['time_metrics']
            print(f"  Val Time F1: {tm.get('f1_score', 0):.4f}, Precision: {tm.get('precision', 0):.4f}, Recall: {tm.get('recall', 0):.4f}")
            if 'auc_roc' in tm:
                print(f"  Val AUC-ROC: {tm['auc_roc']:.4f}, AUC-PR: {tm['auc_pr']:.4f}")
        if 'segment_metrics' in val_metrics:
            sm = val_metrics['segment_metrics']
            print(f"  Val Segment F1: {sm.get('segment_f1', 0):.4f}, Detected: {sm.get('detected_spindles', 0)}/{sm.get('detected_spindles', 0) + sm.get('missed_spindles', 0)}")

    def _save_checkpoint(self, model, optimizer, epoch, training_history, model_name):
        """Save training checkpoint"""
        checkpoint_path = self.config.MODEL_DIR / f"{model_name.lower()}_checkpoint_epoch_{epoch + 1}.pth"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_history': training_history,
            'config': self.config.to_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    def comprehensive_evaluation(self, model, test_loader, model_name="model"):
        """Comprehensive evaluation with multiple thresholds and time-based analysis"""
        print(f"\n{'=' * 60}")
        print("COMPREHENSIVE TIME-BASED EVALUATION")
        print(f"{'=' * 60}")
        # Test multiple thresholds
        test_thresholds = self.config.get('evaluation.test_thresholds', [0.1, 0.2, 0.3, 0.5])
        threshold_results = {}
        print("\nTesting multiple thresholds...")
        for threshold in test_thresholds:
            print(f"\n--- THRESHOLD {threshold:.1f} ---")
            results = self._evaluate_time_based(
                model, test_loader,
                start_time=self.config.TEST_START,
                end_time=self.config.TEST_END,
                threshold=threshold
            )
            threshold_results[threshold] = results
            # Print summary
            if 'time_metrics' in results and 'segment_metrics' in results:
                tm = results['time_metrics']
                sm = results['segment_metrics']
                print(f"Time F1: {tm.get('f1_score', 0):.4f}, Segment F1: {sm.get('segment_f1', 0):.4f}")
        # Find optimal threshold
        print(f"\n--- FINDING OPTIMAL THRESHOLD ---")
        optimal_threshold, threshold_analysis = self.time_metrics.find_optimal_threshold_timeline(
            model, test_loader, self.config.DEVICE,
            self.spindle_annotations, self.config.TEST_START, self.config.TEST_END,
            test_thresholds=np.arange(0.05, 0.95, 0.05)
        )
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        # Final evaluation with optimal threshold
        final_results = self._evaluate_time_based(
            model, test_loader,
            start_time=self.config.TEST_START,
            end_time=self.config.TEST_END,
            threshold=optimal_threshold
        )
        # Detailed report
        self.time_metrics.print_detailed_report(final_results, "Test Set (Optimal Threshold)")
        # Create visualizations
        if self.config.get('visualization.save_plots', True):
            self._create_visualizations(final_results, threshold_analysis, model_name)
        # Log to experiment tracker
        if self.experiment_tracker:
            self.experiment_tracker.log_final_results(final_results, threshold_analysis, optimal_threshold)
        return {
            'final_results': final_results,
            'threshold_results': threshold_results,
            'threshold_analysis': threshold_analysis,
            'optimal_threshold': optimal_threshold
        }

    def _create_visualizations(self, final_results, threshold_analysis, model_name):
        """Create and save visualizations"""
        results_dir = self.config.RESULTS_DIR
        plot_format = self.config.get('visualization.plot_format', 'png')
        # Timeline plot
        if self.config.get('visualization.plot_predictions_timeline', True):
            timeline_path = results_dir / f"{model_name}_timeline.{plot_format}"
            self.time_metrics.plot_timeline(final_results, str(timeline_path))
            if self.experiment_tracker:
                import matplotlib.pyplot as plt
                fig = plt.gcf()
                self.experiment_tracker.logger.log_timeline_plot(fig, "timeline")
        # Threshold analysis plot
        if self.config.get('visualization.plot_threshold_analysis', True):
            threshold_path = results_dir / f"{model_name}_threshold_analysis.{plot_format}"
            self.time_metrics.plot_threshold_analysis(threshold_analysis, str(threshold_path))
            if self.experiment_tracker:
                import matplotlib.pyplot as plt
                fig = plt.gcf()
                self.experiment_tracker.logger.log_timeline_plot(fig, "threshold_analysis")

    def save_final_model(self, model, training_results, model_name="model"):
        """Save final trained model with metadata"""
        model_path = self.config.MODEL_DIR / f"{model_name.lower()}_final.pth"
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'config': self.config.to_dict(),
            'training_results': training_results,
            'spindle_annotations_count': len(self.spindle_annotations),
            'timestamp': time.time()
        }
        torch.save(save_dict, model_path)
        print(f"\nFinal model saved to: {model_path}")
        # Save to experiment tracker
        if self.experiment_tracker:
            self.experiment_tracker.save_experiment(model_path)
        return model_path

    def finish_experiment(self):
        """Clean up experiment tracking"""
        if self.experiment_tracker:
            self.experiment_tracker.finish()
