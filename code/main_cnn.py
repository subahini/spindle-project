"""
 CNN EEG Spindle Detection

"""
import torch
from pathlib import Path



from models import  SpindleCNN , UNet1D
from metrics import CustomMetrics
from data_loader import load_and_preprocess_data, load_spindle_labels, create_windows, get_data_loaders

from trainer import train_model
from config import Config

def main():

    config = Config()

    try:
        print(f"Using device: {config.DEVICE}")
        print("=" * 60)
        print(" CNN EEG SPINDLE DETECTION PIPELINE")
        print("=" * 60)

        # === DATA PREPARATION ===
        print("\n1. LOADING AND PREPROCESSING DATA")
        print("-" * 40)

        # Load and preprocess data
        raw, sfreq = load_and_preprocess_data()
        spindles = load_spindle_labels()

        print(" MODEL: UNet1d , 7% overlap , Focal loss with alph 75_downsample majority class")
        # MY MODEL
        print("\n3. TRAINING THE MODEL")
        print("-" * 40)



       # model = SpindleCNN().to(config.DEVICE)
       # model_name = "SpindleCNN"
        model = UNet1D().to(config.DEVICE)
        model_name = "UNet1D"

        # Create data splits

        print("\n2. CREATING DATA SPLITS")
        print("-" * 40)
        create_windows(raw, sfreq, spindles, config.TRAIN_START, config.TRAIN_END, "train",model_name)   # for 1d and 2 d we nedd dift data set
        create_windows(raw, sfreq, spindles, config.TEST_START, config.TEST_END, "test",model_name)
        create_windows(raw, sfreq, spindles, config.VAL_START, config.VAL_END, "val",model_name)

        # Get data loaders
        train_loader, val_loader, test_loader = get_data_loaders()

        print(f"Selected Model: {model_name}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # === TRAINING ===
        print("\n4. TRAINING PHASE")
        print("-" * 40)
        training_results = train_model(model, train_loader, val_loader, config.DEVICE, model_name)

        # === SAVE MODEL ===
        print("\n5. SAVING MODEL")
        print("-" * 40)
        model_path = config.MODEL_DIR / f"{model_name.lower()}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'config': config.__dict__,
            'training_results': training_results
        }, model_path)
        print(f"Model saved to {model_path}")

        # === COMPREHENSIVE EVALUATION ===
        print("\n6. COMPREHENSIVE EVALUATION")
        print("=" * 60)

        metrics_calc = CustomMetrics()

        # Test with multiple thresholds
        thresholds = [0.1, 0.2, 0.3, 0.5]

        for threshold in thresholds:
            print(f"\n--- EVALUATION WITH THRESHOLD {threshold} ---")
            test_metrics = metrics_calc.evaluate_model_with_threshold(
                model, test_loader, config.DEVICE, threshold=threshold
            )
            metrics_calc.print_detailed_report(test_metrics, f"Test Set (threshold={threshold})")

        # Find and evaluate with optimal threshold
        print("\n--- OPTIMAL THRESHOLD EVALUATION ---")
        optimal_threshold = metrics_calc.find_optimal_threshold()
        print(f"Optimal threshold found: {optimal_threshold:.3f}")

        final_metrics = metrics_calc.evaluate_model_with_threshold(
            model, test_loader, config.DEVICE, threshold=optimal_threshold
        )
        metrics_calc.print_detailed_report(final_metrics, "Test Set (Optimal Threshold)")

        # === VISUALIZATION ===
        print("\n7. GENERATING PLOTS")
        print("-" * 40)

        # Plot all curves and save to results directory
        plot_prefix = f"{model_name.lower()}_"
        metrics_calc.plot_all_curves(final_metrics, config.RESULTS_DIR, plot_prefix)

        print(f"\n{'=' * 60}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"Best Validation F1: {training_results['best_val_f1']:.4f}")
        print(f"Final Test F1: {final_metrics['f1_score']:.4f}")
        print(f"Final Test AUC-ROC: {final_metrics.get('auc_roc', 'N/A')}")
        print(f"Final Test AUC-PR: {final_metrics.get('auc_pr', 'N/A')}")
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        print(f"Training Time: {training_results['total_time']:.1f} seconds")

    except Exception as e:
        print(f"Error in pipeline: {e}")
        raise


if __name__ == "__main__":
    main()