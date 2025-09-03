"""
Weights & Biases integration for EEG Spindle Detection
Handles experiment tracking, hyperparameter logging, and visualization
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import yaml

class WandBLogger:
    """Weights & Biases logging integration"""

    def __init__(self, config):
        self.config = config
        self.wandb_config = config.config['wandb']
        self.enabled = self.wandb_config.get('enabled', True)
        self.run = None

        if self.enabled:
            self._initialize_wandb()

    def _initialize_wandb(self):
        """Initialize wandb run with configuration"""
        try:
            # If wandb run already exists (e.g., within a sweep), reuse it
            if wandb.run is not None:
                self.run = wandb.run
                print("W&B run already active, skipping new initialization.")
                return
            # Initialize wandb
            self.run = wandb.init(
                project=self.wandb_config.get('project', 'eeg-spindle-detection'),
                entity=self.wandb_config.get('entity', None),
                name=self._generate_run_name(),
                notes=self.wandb_config.get('notes', ''),
                tags=self.wandb_config.get('tags', []),
                mode=self.wandb_config.get('mode', 'online'),
                config=self._prepare_config_for_wandb()
            )

            print(f"W&B run initialized: {self.run.name}")
            print(f"W&B URL: {self.run.url}")

        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            self.enabled = False

    def _generate_run_name(self):
        """Generate descriptive run name"""
        model_name = self.config.MODEL_NAME
        loss_func = self.config.get('training.loss_function', 'bce')
        lr = self.config.LEARNING_RATE
        return f"{model_name}_{loss_func}_lr{lr}"

    def _prepare_config_for_wandb(self):
        """Prepare configuration dictionary for wandb"""
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_config = flatten_dict(self.config.to_dict())
        # Add computed properties
        flat_config.update({
            'device': str(self.config.DEVICE),
            'num_train_hours': self.config.get('data.train_end') - self.config.get('data.train_start'),
            'num_val_hours': self.config.get('data.val_end') - self.config.get('data.val_start'),
            'num_test_hours': self.config.get('data.test_end') - self.config.get('data.test_start'),
            'window_overlap': 1 - (self.config.STEP_SEC / self.config.WINDOW_SEC),
        })
        return flat_config

    def log_metrics(self, metrics_dict: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb"""
        if not self.enabled:
            return
        try:
            wandb.log(metrics_dict, step=step)
        except Exception as e:
            print(f"Warning: Failed to log metrics to wandb: {e}")

    def log_model_architecture(self, model):
        """Log model architecture and parameters"""
        if not self.enabled:
            return
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            wandb.log({
                'model/total_parameters': total_params,
                'model/trainable_parameters': trainable_params,
                'model/model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
            })
            model_summary = str(model)
            wandb.log({"model/architecture": wandb.Html(f"<pre>{model_summary}</pre>")})
        except Exception as e:
            print(f"Warning: Failed to log model architecture: {e}")

    def log_training_step(self, epoch: int, step: int, loss: float, metrics: Optional[Dict] = None):
        """Log training step metrics"""
        if not self.enabled:
            return
        log_dict = {
            'train/loss': loss,
            'train/epoch': epoch,
            'train/step': step
        }
        if metrics:
            for key, value in metrics.items():
                log_dict[f'train/{key}'] = value
        self.log_metrics(log_dict, step=step)

    def log_validation_metrics(self, epoch: int, val_metrics: Dict[str, Any]):
        """Log validation metrics"""
        if not self.enabled:
            return
        log_dict = {'val/epoch': epoch}
        for key, value in val_metrics.items():
            if isinstance(value, (int, float, np.number)):
                log_dict[f'val/{key}'] = float(value)
        self.log_metrics(log_dict)

    def log_test_results(self, test_results: Dict[str, Any], threshold: float):
        """Log comprehensive test results"""
        if not self.enabled:
            return
        log_dict = {'test/threshold': threshold}
        if 'time_metrics' in test_results:
            tm = test_results['time_metrics']
            for key, value in tm.items():
                if isinstance(value, (int, float, np.number)):
                    log_dict[f'test/time_{key}'] = float(value)
        if 'segment_metrics' in test_results:
            sm = test_results['segment_metrics']
            for key, value in sm.items():
                if isinstance(value, (int, float, np.number)):
                    log_dict[f'test/segment_{key}'] = float(value)
        if 'target_segments' in test_results and 'predicted_segments' in test_results:
            log_dict.update({
                'test/total_true_spindles': len(test_results['target_segments']),
                'test/total_predicted_segments': len(test_results['predicted_segments']),
                'test/total_spindle_time': sum(e - s for s, e in test_results['target_segments']),
                'test/total_predicted_time': sum(e - s for s, e in test_results['predicted_segments'])
            })
        self.log_metrics(log_dict)

    def log_threshold_analysis(self, threshold_results: Dict[float, Dict]):
        """Log threshold analysis results"""
        if not self.enabled:
            return
        table_data = []
        for threshold, results in sorted(threshold_results.items()):
            row = [
                threshold,
                results['time_f1'],
                results['segment_f1'],
                results['combined_f1'],
                results['time_metrics']['precision'],
                results['time_metrics']['recall']
            ]
            table_data.append(row)
        table = wandb.Table(columns=['threshold', 'time_f1', 'segment_f1', 'combined_f1', 'precision', 'recall'], data=table_data)
        wandb.log({'test/threshold_analysis': table})

    def log_timeline_plot(self, fig, name: str = "timeline"):
        """Log matplotlib timeline plot"""
        if not self.enabled:
            return
        try:
            wandb.log({f"plots/{name}": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Failed to log plot {name}: {e}")

    def log_confusion_matrix(self, confusion_matrix, class_names=None):
        """Log confusion matrix visualization"""
        if not self.enabled:
            return
        try:
            if class_names is None:
                class_names = ['No Spindle', 'Spindle']
            wandb.log({
                "plots/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=[0, 0, 1, 1],
                    preds=[0, 1, 0, 1],
                    class_names=class_names
                )
            })
        except Exception as e:
            print(f"Warning: Failed to log confusion matrix: {e}")

    def log_learning_curves(self, train_losses: List[float], val_f1_scores: List[float]):
        """Log learning curves"""
        if not self.enabled:
            return
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(train_losses)
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
            ax2.plot(val_f1_scores)
            ax2.set_title('Validation F1 Score')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('F1 Score')
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            wandb.log({"plots/learning_curves": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Failed to log learning curves: {e}")

    def log_hyperparameter_importance(self, param_importance: Dict[str, float]):
        """Log hyperparameter importance analysis"""
        if not self.enabled:
            return
        try:
            params = list(param_importance.keys())
            importance = list(param_importance.values())
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(params, importance)
            ax.set_xlabel('Importance')
            ax.set_title('Hyperparameter Importance')
            plt.tight_layout()
            wandb.log({"plots/hyperparameter_importance": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Failed to log hyperparameter importance: {e}")

    def log_model_predictions(self, predictions_sample: Dict[str, Any]):
        """Log sample predictions for inspection"""
        if not self.enabled:
            return
        try:
            table_data = []
            for i, (pred, target, prob) in enumerate(zip(predictions_sample['predictions'][:100], predictions_sample['targets'][:100], predictions_sample['probabilities'][:100])):
                table_data.append([i, float(pred), float(target), float(prob)])
            table = wandb.Table(columns=['sample_id', 'prediction', 'target', 'probability'], data=table_data)
            wandb.log({'predictions/sample_predictions': table})
        except Exception as e:
            print(f"Warning: Failed to log model predictions: {e}")

    def log_dataset_statistics(self, dataset_stats: Dict[str, Any]):
        """Log dataset statistics"""
        if not self.enabled:
            return
        log_dict = {}
        for split, stats in dataset_stats.items():
            for key, value in stats.items():
                if isinstance(value, (int, float, np.number)):
                    log_dict[f'data/{split}_{key}'] = float(value)
        self.log_metrics(log_dict)

    def save_model_artifact(self, model_path: Path, model_name: str = "model"):
        """Save model as wandb artifact"""
        if not self.enabled:
            return
        try:
            artifact = wandb.Artifact(
                name=f"{model_name}_{self.run.id}",
                type="model",
                description=f"Trained {self.config.MODEL_NAME} model"
            )
            artifact.add_file(str(model_path))
            wandb.log_artifact(artifact)
            print(f"Model saved as W&B artifact: {artifact.name}")
        except Exception as e:
            print(f"Warning: Failed to save model artifact: {e}")

    def log_config_file(self, config_path: str):
        """Log configuration file as artifact"""
        if not self.enabled:
            return
        try:
            artifact = wandb.Artifact(
                name=f"config_{self.run.id}",
                type="config",
                description="Experiment configuration"
            )
            artifact.add_file(config_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"Warning: Failed to log config artifact: {e}")

    def finish(self):
        """Finish wandb run"""
        if self.enabled and self.run:
            wandb.finish()
            print("W&B run finished")

    def watch_model(self, model, log_freq: int = 1000):
        """Watch model gradients and parameters"""
        if not self.enabled:
            return
        try:
            wandb.watch(
                model,
                log="all" if self.config.get('logging.log_gradients', False) else "parameters",
                log_freq=log_freq
            )
        except Exception as e:
            print(f"Warning: Failed to watch model: {e}")

    def alert(self, title: str, text: str, level: str = "INFO"):
        """Send wandb alert"""
        if not self.enabled:
            return
        try:
            wandb.alert(title=title, text=text, level=level)
        except Exception as e:
            print(f"Warning: Failed to send alert: {e}")

class ExperimentTracker:
    """High-level experiment tracking wrapper"""

    def __init__(self, config):
        self.config = config
        self.logger = WandBLogger(config)
        self.training_history = {
            'train_losses': [],
            'val_metrics': [],
            'epochs': []
        }

    def start_experiment(self, model):
        """Start experiment tracking"""
        print("Starting experiment tracking...")
        self.logger.log_model_architecture(model)
        self.logger.watch_model(model)

    def log_epoch(self, epoch: int, train_loss: float, val_metrics: Dict[str, Any]):
        """Log epoch results"""
        self.training_history['train_losses'].append(train_loss)
        self.training_history['val_metrics'].append(val_metrics)
        self.training_history['epochs'].append(epoch)
        self.logger.log_training_step(epoch, epoch, train_loss)
        self.logger.log_validation_metrics(epoch, val_metrics)

    def log_final_results(self, test_results: Dict[str, Any], threshold_analysis: Dict, best_threshold: float):
        """Log final experiment results"""
        self.logger.log_test_results(test_results, best_threshold)
        self.logger.log_threshold_analysis(threshold_analysis)
        # Log learning curves
        val_f1s = [m.get('f1_score', 0) for m in self.training_history['val_metrics']]
        self.logger.log_learning_curves(self.training_history['train_losses'], val_f1s)
        # Alert for experiment completion
        f1_score = test_results.get('time_metrics', {}).get('f1_score', 0)
        self.logger.alert(
            title="Experiment Completed",
            text=f"Model: {self.config.MODEL_NAME}, Best F1: {f1_score:.4f}",
            level="INFO"
        )

    def save_experiment(self, model_path: Path):
        """Save experiment artifacts"""
        self.logger.save_model_artifact(model_path)
        # Save configuration
        config_path = model_path.parent / "experiment_config.yaml"
        self.config.save(str(config_path))
        self.logger.log_config_file(str(config_path))

    def finish(self):
        """Finish experiment tracking"""
        self.logger.finish()
