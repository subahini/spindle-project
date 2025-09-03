"""
Main script for time-based EEG Spindle Detection with wandb integration
Run with: python main_cnn.py --model SpindleCNN --loss focal --threshold 0.3
"""

import torch
import sys
import traceback
import numpy as np
from pathlib import Path

from models import SpindleCNN, UNet1D
from data_loader import load_and_preprocess_data, load_spindle_labels, create_windows, get_data_loaders
from trainer import SpindleTrainer
from metrics import TimeBasedMetrics
from config import Config

def setup_reproducibility(seed=42):
    """Setup reproducibility for experiments"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(config=None, config_path="config.yaml"):
    """Main pipeline for time-based spindle detection"""
    try:
        # Load configuration (from object or file)
        if config is None:
            config = Config(config_path)
        # Setup reproducibility
        setup_reproducibility(config.get('seed', 42))
        print("=" * 80)
        print("TIME-BASED EEG SPINDLE DETECTION PIPELINE")
        print("=" * 80)
        print(f"Using device: {config.DEVICE}")
        print(f"Model: {config.MODEL_NAME}")
        print(f"Loss function: {config.get('training.loss_function', 'bce')}")
        print(f"Time resolution: {config.TIME_RESOLUTION}s")
        print(f"WandB enabled: {config.get('wandb.enabled', True)}")
        # === DATA PREPARATION ===
        print("\n" + "=" * 50)
        print("1. DATA PREPARATION")
        print("=" * 50)
        print("Loading and preprocessing EEG data...")
        raw, sfreq = load_and_preprocess_data(config)
        spindles = load_spindle_labels(config)
        print(f"Loaded {len(spindles)} spindle annotations")
        print(f"Sampling frequency: {sfreq} Hz")
        print(f"EEG channels: {len(config.EEG_CHANNELS)}")

        rec_sec = float(raw.times[-1])
        print(f"[DEBUG] EDF duration: {rec_sec:.2f}s ({rec_sec / 3600:.2f}h), sfreq={sfreq} Hz")
        print(f"[DEBUG] SPLITS (hours) -> "
              f"Train {config.get('data.train_start')}–{config.get('data.train_end')}, "
              f"Val {config.get('data.val_start')}–{config.get('data.val_end')}, "
              f"Test {config.get('data.test_start')}–{config.get('data.test_end')}")

        # === DATA SPLITS ===
        print("\n2. CREATING DATA SPLITS")
        print("=" * 50)
        model_name = config.MODEL_NAME
        print(f"Creating time-based data splits for {model_name}...")
        print(f"Train: {config.get('data.train_start'):.1f}h - {config.get('data.train_end'):.1f}h")
        print(f"Val:   {config.get('data.val_start'):.1f}h - {config.get('data.val_end'):.1f}h")
        print(f"Test:  {config.get('data.test_start'):.1f}h - {config.get('data.test_end'):.1f}h")
        create_windows(raw, sfreq, spindles, config.TRAIN_START, config.TRAIN_END, "train", model_name, config)
        create_windows(raw, sfreq, spindles, config.VAL_START, config.VAL_END, "val", model_name, config)
        create_windows(raw, sfreq, spindles, config.TEST_START, config.TEST_END, "test", model_name, config)
        train_loader, val_loader, test_loader = get_data_loaders(model_type=model_name, config=config)
          # debugging ?? why no result

        # === DEBUG: dataset sanity checks ===
        import numpy as np

        print("\n[DEBUG] Verifying saved arrays and loader sizes...")
        sd = config.SAVE_DIR
        shapes = {}
        for split in ("train", "val", "test"):
            x_path = sd / f"X_{split}.npy"
            y_path = sd / f"y_{split}.npy"
            print(f"[DEBUG] {split}: X={x_path.exists()} Y={y_path.exists()} -> {x_path} , {y_path}")
            if x_path.exists():
                try:
                    x = np.load(x_path, mmap_mode="r")
                    shapes[f"X_{split}"] = x.shape
                except Exception as e:
                    print(f"[DEBUG] failed to load {x_path}: {e}")
            if y_path.exists():
                try:
                    y = np.load(y_path, mmap_mode="r")
                    shapes[f"y_{split}"] = y.shape
                except Exception as e:
                    print(f"[DEBUG] failed to load {y_path}: {e}")

        print(f"[DEBUG] shapes: {shapes}")
        print(f"[DEBUG] loader sizes -> train batches={len(train_loader)}, "
              f"val batches={len(val_loader)}, test batches={len(test_loader)}")

        # Stop early if no training data
        if len(train_loader) == 0:
            raise RuntimeError("[DEBUG] Train loader has 0 batches. Check create_windows/train split.")

        print("Data splits created successfully")
        overlap_pct = (1 - config.STEP_SEC / config.WINDOW_SEC) * 100
        print(f"Window duration: {config.WINDOW_SEC}s, Step: {config.STEP_SEC}s (overlap: {overlap_pct:.1f}%)")
        # === MODEL SETUP ===
        print("\n3. MODEL SETUP")
        print("=" * 50)
        # Initialize model
        if model_name == "SpindleCNN":
            model = SpindleCNN(n_channels=len(config.EEG_CHANNELS), dropout_rate=config.get('model.dropout_rate', 0.4)).to(config.DEVICE)
        elif model_name == "UNet1D":
            model = UNet1D(in_channels=len(config.EEG_CHANNELS), init_features=config.get('model.init_features', 64)).to(config.DEVICE)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: ~{total_params * 4 / (1024 ** 2):.1f} MB")
        # === TRAINER SETUP ===
        print("\n4. TRAINING SETUP")
        print("=" * 50)
        trainer = SpindleTrainer(config, spindles)
        print("Training configuration:")
        print(f"  Epochs: {config.EPOCHS}")
        print(f"  Batch size: {config.BATCH_SIZE}")
        print(f"  Learning rate: {config.LEARNING_RATE}")
        print(f"  Optimizer: {config.get('training.optimizer', 'adam')}")
        print(f"  Loss function: {config.get('training.loss_function', 'bce')}")
        print(f"  Early stopping patience: {config.PATIENCE}")
        if config.get('training.loss_function') == 'focal':
            print(f"  Focal loss alpha: {config.get('training.focal_alpha', 0.25)}")
            print(f"  Focal loss gamma: {config.get('training.focal_gamma', 2.0)}")
        # === TRAINING ===
        print("\n5. TRAINING PHASE")
        print("=" * 50)
        training_results = trainer.train_model(model=model, train_loader=train_loader, val_loader=val_loader, model_name=model_name)
        print("\nTraining completed!")
        print(f"Best validation F1: {training_results['training_history']['best_val_f1']:.4f}")
        print(f"Training time: {training_results['training_history']['total_time']:.1f} seconds")
        # === COMPREHENSIVE EVALUATION ===
        print("\n6. COMPREHENSIVE TIME-BASED EVALUATION")
        print("=" * 50)
        evaluation_results = trainer.comprehensive_evaluation(model=training_results['model'], test_loader=test_loader, model_name=model_name)
        # === SAVE MODEL ===
        print("\n7. SAVING RESULTS")
        print("=" * 50)
        model_path = trainer.save_final_model(model=training_results['model'], training_results={**training_results['training_history'], **evaluation_results}, model_name=model_name)
        # === FINAL SUMMARY ===
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        final_results = evaluation_results['final_results']
        optimal_threshold = evaluation_results['optimal_threshold']
        print(f"Model: {model_name}")
        print(f"Parameters: {total_params:,}")
        print(f"Training time: {training_results['training_history']['total_time']:.1f}s")
        print(f"Best validation F1: {training_results['training_history']['best_val_f1']:.4f}")
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        if 'time_metrics' in final_results:
            tm = final_results['time_metrics']
            print(f"Final time-based F1: {tm.get('f1_score', 0):.4f}")
            print(f"Final time-based precision: {tm.get('precision', 0):.4f}")
            print(f"Final time-based recall: {tm.get('recall', 0):.4f}")
            if 'auc_roc' in tm:
                print(f"Final AUC-ROC: {tm['auc_roc']:.4f}")
                print(f"Final AUC-PR: {tm['auc_pr']:.4f}")
        if 'segment_metrics' in final_results:
            sm = final_results['segment_metrics']
            print(f"Final segment-based F1: {sm.get('segment_f1', 0):.4f}")
            print(f"Spindles detected: {sm.get('detected_spindles', 0)}/{len(spindles)}")
        print(f"\nResults saved to: {config.RESULTS_DIR}")
        print(f"Model saved to: {model_path}")
        print("\n--- Configuration Summary ---")
        print(f"Config file: {config.config_path}")
        print(f"Command line args applied: {' '.join(sys.argv[1:])}")
        print(f"Random seed: {config.get('seed', 42)}")
        # Performance insights
        print("\n--- Performance Insights ---")
        if 'target_segments' in final_results and 'predicted_segments' in final_results:
            total_spindle_time = sum(e - s for s, e in final_results['target_segments'])
            predicted_time = sum(e - s for s, e in final_results['predicted_segments'])
            analysis_duration = final_results['time_grid'][-1] - final_results['time_grid'][0]
            print(f"Analysis duration: {analysis_duration / 60:.1f} minutes")
            print(f"True spindle density: {len(final_results['target_segments']) / (analysis_duration / 60):.1f} spindles/min")
            print(f"True spindle coverage: {total_spindle_time / analysis_duration * 100:.1f}% of time")
            print(f"Predicted coverage: {predicted_time / analysis_duration * 100:.1f}% of time")
        return {
            'model': training_results['model'],
            'config': config,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'model_path': model_path
        }
    except Exception as e:
        print("\nERROR: Pipeline failed with exception:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        raise e
    finally:
        try:
            if 'trainer' in locals():
                trainer.finish_experiment()
        except:
            pass

def run_hyperparameter_sweep(config_path="config.yaml"):
    """Run hyperparameter sweep using wandb"""
    import wandb
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'test/time_f1_score', 'goal': 'maximize'},
        'parameters': {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 0.0001, 'max': 0.01},
            'batch_size': {'values': [16, 32, 64, 128]},
            'dropout_rate': {'distribution': 'uniform', 'min': 0.1, 'max': 0.6},
            'focal_alpha': {'distribution': 'uniform', 'min': 0.1, 'max': 0.9},
            'focal_gamma': {'distribution': 'uniform', 'min': 1.0, 'max': 5.0},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 0.00001, 'max': 0.001}
        }
    }
    def sweep_train():
        """Single training run for sweep"""
        with wandb.init() as run:
            config = Config(config_path)
            config.set('training.learning_rate', wandb.config.learning_rate)
            config.set('training.batch_size', wandb.config.batch_size)
            config.set('model.dropout_rate', wandb.config.dropout_rate)
            config.set('training.focal_alpha', wandb.config.focal_alpha)
            config.set('training.focal_gamma', wandb.config.focal_gamma)
            config.set('training.weight_decay', wandb.config.weight_decay)
            results = main(config=config)
            final_f1 = results['evaluation_results']['final_results']['time_metrics']['f1_score']
            wandb.log({'final_f1': final_f1})
    sweep_id = wandb.sweep(sweep_config, project="eeg-spindle-sweep")
    wandb.agent(sweep_id, sweep_train)

def run_ablation_study(config_path="config.yaml"):
    """Run ablation study on different configurations"""
    config = Config(config_path)
    ablation_configs = [
        {'name': 'baseline', 'loss': 'bce', 'model': 'SpindleCNN'},
        {'name': 'focal_loss', 'loss': 'focal', 'model': 'SpindleCNN'},
        {'name': 'weighted_bce', 'loss': 'weighted_bce', 'model': 'SpindleCNN'},
        {'name': 'unet_baseline', 'loss': 'bce', 'model': 'UNet1D'},
        {'name': 'unet_focal', 'loss': 'focal', 'model': 'UNet1D'}
    ]
    results = {}
    for ablation in ablation_configs:
        print("\n" + "=" * 80)
        print(f"RUNNING ABLATION: {ablation['name'].upper()}")
        print("=" * 80)
        config.set('training.loss_function', ablation['loss'])
        config.set('model.name', ablation['model'])
        config.set('wandb.tags', ['ablation', ablation['name']])
        try:
            result = main(config=config)
            results[ablation['name']] = {
                'time_f1': result['evaluation_results']['final_results']['time_metrics']['f1_score'],
                'segment_f1': result['evaluation_results']['final_results']['segment_metrics']['segment_f1'],
                'config': ablation
            }
            print(f"Ablation {ablation['name']} completed successfully")
        except Exception as e:
            print(f"Ablation {ablation['name']} failed: {e}")
            results[ablation['name']] = {'error': str(e)}
    # Print ablation summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    for name, result in results.items():
        if 'error' not in result:
            print(f"{name:15s}: Time F1={result['time_f1']:.4f}, Segment F1={result['segment_f1']:.4f}")
        else:
            print(f"{name:15s}: FAILED - {result['error']}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='EEG Spindle Detection Pipeline')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    args, _ = parser.parse_known_args()
    if args.sweep:
        print("Running hyperparameter sweep...")
        run_hyperparameter_sweep(args.config)
    elif args.ablation:
        print("Running ablation study...")
        run_ablation_study(args.config)
    else:
        print("Running single training experiment...")
        main(config_path=args.config)
