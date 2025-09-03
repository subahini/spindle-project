# config.py (refactored)
import yaml
import argparse
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import os

class Config:
    """Configuration manager with YAML support and command line overrides"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self._load_config()
        self._setup_paths()
        self._setup_device()

    def _load_config(self):
        """Load configuration from YAML file"""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        # If wandb mode is "disabled", disable WandB logging
        if 'wandb' in self.config:
            if self.config['wandb'].get('mode') == 'disabled':
                self.config['wandb']['enabled'] = False

        # Apply command line overrides
        self._apply_cli_overrides()

    def _apply_cli_overrides(self):
        """Apply command line argument overrides"""
        parser = argparse.ArgumentParser(description='EEG Spindle Detection')

        # Model arguments
        parser.add_argument('--model', type=str, help='Model name (SpindleCNN or UNet1D)')
        parser.add_argument('--loss', type=str, help='Loss function to use (bce, focal, weighted_bce)')
        parser.add_argument('--threshold', type=float, help='Default classification threshold')
        parser.add_argument('--dropout-rate', type=float, help='Dropout rate for the model')

        # Training arguments
        parser.add_argument('--epochs', type=int, help='Number of training epochs')
        parser.add_argument('--batch-size', type=int, help='Batch size for training')
        parser.add_argument('--lr', type=float, help='Learning rate')
        parser.add_argument('--patience', type=int, help='Early stopping patience')
        parser.add_argument('--weight-decay', type=float, help='Weight decay (L2 regularization)')
        parser.add_argument('--optimizer', type=str, help='Optimizer type (adam, sgd, adamw)')
        parser.add_argument('--scheduler', type=str, help='LR scheduler type (plateau, step, cosine)')

        # Data split arguments
        parser.add_argument('--train-hours', type=str, help='Training data hours (start,end)')
        parser.add_argument('--val-hours', type=str, help='Validation data hours (start,end)')
        parser.add_argument('--test-hours', type=str, help='Test data hours (start,end)')

        # Focal loss arguments
        parser.add_argument('--focal-alpha', type=float, help='Focal loss alpha parameter')
        parser.add_argument('--focal-gamma', type=float, help='Focal loss gamma parameter')

        # WandB arguments
        parser.add_argument('--wandb-project', type=str, help='Weights & Biases project name')
        parser.add_argument('--wandb-disabled', action='store_true', help='Disable Weights & Biases logging')
        parser.add_argument('--wandb-offline', action='store_true', help='Run Weights & Biases in offline mode')

        # Device argument
        parser.add_argument('--device', type=str, help='Computation device to use (auto, cpu, cuda, mps)')

        # Parse only the known arguments
        args, _ = parser.parse_known_args()

        # Override configuration with CLI arguments if provided
        if args.model:
            self.config['model']['name'] = args.model
        if args.loss:
            self.config['training']['loss_function'] = args.loss
        if args.threshold is not None:
            self.config['evaluation']['default_threshold'] = args.threshold
        if args.dropout_rate is not None:
            self.config['model']['dropout_rate'] = args.dropout_rate

        if args.epochs:
            self.config['training']['epochs'] = args.epochs
        if args.batch_size:
            self.config['training']['batch_size'] = args.batch_size
        if args.lr:
            self.config['training']['learning_rate'] = args.lr
        if args.patience:
            self.config['training']['patience'] = args.patience
        if args.weight_decay is not None:
            self.config['training']['weight_decay'] = args.weight_decay
        if args.optimizer:
            self.config['training']['optimizer'] = args.optimizer
        if args.scheduler:
            self.config['training']['scheduler'] = args.scheduler

        if args.focal_alpha:
            self.config['training']['focal_alpha'] = args.focal_alpha
        if args.focal_gamma:
            self.config['training']['focal_gamma'] = args.focal_gamma

        if args.wandb_project:
            self.config['wandb']['project'] = args.wandb_project
        if args.wandb_disabled:
            self.config['wandb']['enabled'] = False
        if args.wandb_offline:
            self.config['wandb']['mode'] = 'offline'

        if args.device:
            self.config['hardware']['device'] = args.device

        # Handle time range overrides for data splits (in hours)
        if args.train_hours:
            start, end = map(float, args.train_hours.split(','))
            self.config['data']['train_start'] = start
            self.config['data']['train_end'] = end
        if args.val_hours:
            start, end = map(float, args.val_hours.split(','))
            self.config['data']['val_start'] = start
            self.config['data']['val_end'] = end
        if args.test_hours:
            start, end = map(float, args.test_hours.split(','))
            self.config['data']['test_start'] = start
            self.config['data']['test_end'] = end

    def _setup_paths(self):
        """Setup directory paths for saving data, results, and models"""
        for path_key in ['save_dir', 'results_dir', 'model_dir']:
            path = Path(self.config['data'][path_key])
            path.mkdir(parents=True, exist_ok=True)
            setattr(self, path_key.upper(), path)

        # File paths for raw data and labels
        self.EDF_PATH = Path(self.config['data']['edf_path'])
        self.JSON_PATH = Path(self.config['data']['json_path'])

        # Ensure log directory exists
        log_file = Path(self.config['logging']['log_file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)

    def _setup_device(self):
        """Setup computation device (CPU/CUDA/MPS)"""
        device_config = self.config['hardware']['device']
        if device_config == 'auto':
            if torch.cuda.is_available():
                self.DEVICE = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.DEVICE = torch.device('mps')
            else:
                self.DEVICE = torch.device('cpu')
        else:
            self.DEVICE = torch.device(device_config)

    def get(self, key_path: str, default=None):
        """Get a nested configuration value using dot notation (e.g. 'training.epochs')"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value):
        """Set a nested configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config or not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from a dictionary, deep merging into existing config"""
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        deep_update(self.config, updates)
        # If hardware device was updated, recompute the device selection
        self._setup_device()

    def save(self, path: Optional[str] = None):
        """Save the current configuration to a YAML file"""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    # Convenience properties for easy access and backward compatibility
    @property
    def MODEL_NAME(self):
        return self.config['model']['name']

    @property
    def EPOCHS(self):
        return self.config['training']['epochs']

    @property
    def BATCH_SIZE(self):
        return self.config['training']['batch_size']

    @property
    def LEARNING_RATE(self):
        return self.config['training']['learning_rate']

    @property
    def WEIGHT_DECAY(self):
        return self.config['training']['weight_decay']

    @property
    def PATIENCE(self):
        return self.config['training']['patience']

    @property
    def EEG_CHANNELS(self):
        return self.config['data']['eeg_channels']

    @property
    def FILTER_LOW(self):
        return self.config['preprocessing']['filter_low']

    @property
    def FILTER_HIGH(self):
        return self.config['preprocessing']['filter_high']

    @property
    def WINDOW_SEC(self):
        return self.config['preprocessing']['window_sec']

    @property
    def STEP_SEC(self):
        return self.config['preprocessing']['step_sec']

    @property
    def OVERLAP_THRESHOLD(self):
        return self.config['preprocessing']['overlap_threshold']

    @property
    def DOWNSAMPLE(self):
        return self.config['preprocessing']['downsample_majority']

    @property
    def TRAIN_START(self):
        # Convert hours to seconds for convenience
        return self.config['data']['train_start'] * 3600

    @property
    def TRAIN_END(self):
        return self.config['data']['train_end'] * 3600

    @property
    def VAL_START(self):
        return self.config['data']['val_start'] * 3600

    @property
    def VAL_END(self):
        return self.config['data']['val_end'] * 3600

    @property
    def TEST_START(self):
        return self.config['data']['test_start'] * 3600

    @property
    def TEST_END(self):
        return self.config['data']['test_end'] * 3600

    @property
    def TIME_RESOLUTION(self):
        return self.config['preprocessing']['time_resolution']

    @property
    def SMOOTHING_WINDOW(self):
        return self.config['preprocessing']['smoothing_window']

    @property
    def MIN_SPINDLE_DURATION(self):
        return self.config['preprocessing']['min_spindle_duration']

    @property
    def MAX_SPINDLE_GAP(self):
        return self.config['preprocessing']['max_spindle_gap']

    def to_dict(self):
        """Return the configuration as a dictionary"""
        return self.config.copy()

    def __str__(self):
        """String representation of the configuration (YAML format)"""
        return yaml.dump(self.config, default_flow_style=False, indent=2)
