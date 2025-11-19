"""
Main entry point for Deep Learning (CNN) Deforestation Detection Pipeline
Cà Mau Province - Sentinel-1 & Sentinel-2
Patch-based 2D CNN with spatial context

Usage:
    python main_dl.py
"""

import sys
import argparse
import time
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import common modules
from config import (
    DL_CONFIG, DL_OUTPUT_FILES, FEATURE_NAMES,
    create_output_directories, verify_input_files
)
from core.data_loader import DataLoader
from core.feature_extraction import FeatureExtraction

# Import Deep Learning modules
from models.cnn.patch_extractor import PatchExtractor
from models.cnn.spatial_split import SpatialSplitter
from models.cnn.architecture import create_model
from models.cnn.trainer import CNNTrainer
from models.cnn.predictor import RasterPredictor


class DeforestationCNNPipeline:
    """
    Main pipeline orchestrator for CNN-based deforestation detection
    """

    def __init__(self, config: dict = None):
        """
        Initialize pipeline

        Args:
            config: Optional custom configuration (overrides DL_CONFIG)
        """
        self.config = DL_CONFIG.copy()
        if config is not None:
            self.config.update(config)

        self.execution_times = {}
        self.results = {}

    def run(self):
        """Execute the full CNN pipeline"""
        total_start = time.time()

        print("\n" + "="*70)
        print("CNN DEFORESTATION DETECTION PIPELINE")
        print("Patch-based 2D CNN with Spatial Context")
        print("="*70 + "\n")

        # Print configuration
        self._print_config()
        create_output_directories()
        verify_input_files()

        # Step 1-2: Load Data
        print("\n" + "="*70)
        print("STEP 1-2: LOADING DATA")
        print("="*70)
        step_start = time.time()

        loader = DataLoader()
        s2_before, s2_after = loader.load_sentinel2()
        s1_before, s1_after = loader.load_sentinel1()
        ground_truth = loader.load_ground_truth()
        boundary = loader.load_boundary()

        data = {
            's2_before': s2_before,
            's2_after': s2_after,
            's1_before': s1_before,
            's1_after': s1_after,
            'ground_truth': ground_truth,
            'boundary': boundary,
            'metadata': loader.metadata
        }

        self.execution_times['load_data'] = time.time() - step_start
        print(f"\n✓ Data loading completed in {self.execution_times['load_data']:.2f} seconds")

        # Step 3: Feature Extraction
        print("\n" + "="*70)
        print("STEP 3: FEATURE EXTRACTION")
        print("="*70)
        step_start = time.time()

        extractor = FeatureExtraction()
        feature_stack, valid_mask = extractor.extract_features(
            data['s2_before'], data['s2_after'],
            data['s1_before'], data['s1_after']
        )

        self.execution_times['feature_extraction'] = time.time() - step_start
        print(f"\n✓ Feature extraction completed in {self.execution_times['feature_extraction']:.2f} seconds")

        # Step 4: Spatial-Aware Data Splitting
        print("\n" + "="*70)
        print("STEP 4: SPATIAL-AWARE DATA SPLITTING")
        print("="*70)
        step_start = time.time()

        # Generate random seed from timestamp for different splits each run
        random_seed = int(time.time() * 1000) % (2**32)
        print(f"Using random seed: {random_seed}")

        splitter = SpatialSplitter(
            cluster_distance=self.config['cluster_distance'],
            train_size=self.config['train_size'],
            val_size=self.config['val_size'],
            test_size=self.config['test_size'],
            random_state=random_seed
        )

        train_indices, val_indices, test_indices, split_metadata = splitter.spatial_split(
            data['ground_truth'],
            stratify_by_class=self.config['stratify_by_class'],
            verify=True
        )

        self.execution_times['spatial_split'] = time.time() - step_start
        print(f"\n✓ Spatial splitting completed in {self.execution_times['spatial_split']:.2f} seconds")

        # Step 5: Extract Patches at Ground Truth Points
        print("\n" + "="*70)
        print("STEP 5: EXTRACT PATCHES")
        print("="*70)
        step_start = time.time()

        patch_extractor = PatchExtractor(patch_size=self.config['patch_size'])
        all_patches, all_labels, valid_gt_indices = patch_extractor.extract_patches_at_points(
            feature_stack,
            data['ground_truth'],
            data['metadata']['s2_before']['transform'],
            valid_mask
        )

        # Normalize patches
        patch_extractor.normalize_patches(method=self.config['normalize_method'])

        # Map original indices to patch indices
        index_mapping = {orig_idx: patch_idx for patch_idx, orig_idx in enumerate(valid_gt_indices)}

        # Get patches for each split
        train_patch_indices = [index_mapping[i] for i in train_indices if i in index_mapping]
        val_patch_indices = [index_mapping[i] for i in val_indices if i in index_mapping]
        test_patch_indices = [index_mapping[i] for i in test_indices if i in index_mapping]

        X_train = all_patches[train_patch_indices]
        y_train = all_labels[train_patch_indices]
        X_val = all_patches[val_patch_indices]
        y_val = all_labels[val_patch_indices]
        X_test = all_patches[test_patch_indices]
        y_test = all_labels[test_patch_indices]

        print(f"\nPatch splits:")
        print(f"  Train: {len(X_train)} patches")
        print(f"  Val:   {len(X_val)} patches")
        print(f"  Test:  {len(X_test)} patches")

        # Save patches (optional)
        np.savez_compressed(
            DL_OUTPUT_FILES['training_patches'],
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )
        print(f"✓ Patches saved to: {DL_OUTPUT_FILES['training_patches']}")

        self.execution_times['extract_patches'] = time.time() - step_start
        print(f"\n✓ Patch extraction completed in {self.execution_times['extract_patches']:.2f} seconds")

        # Step 6: Create and Train CNN Model
        print("\n" + "="*70)
        print("STEP 6: TRAIN CNN MODEL")
        print("="*70)
        step_start = time.time()

        # Create model
        model = create_model(
            model_type=self.config['model_type'],
            patch_size=self.config['patch_size'],
            n_features=self.config['n_features'],
            n_classes=self.config['n_classes'],
            dropout_rate=self.config['dropout_rate']
        )
        print(model.get_model_summary())

        # Calculate class weights if requested
        class_weights = None
        if self.config['use_class_weights']:
            unique, counts = np.unique(y_train, return_counts=True)
            class_weights = [len(y_train) / (len(unique) * c) for c in counts]
            print(f"\nClass weights: {class_weights}")

        # Create trainer
        trainer = CNNTrainer(
            model=model,
            device=self.config['device'],
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            class_weights=class_weights,
            use_lr_scheduler=self.config['use_lr_scheduler'],
            lr_scheduler_patience=self.config['lr_scheduler_patience'],
            lr_scheduler_factor=self.config['lr_scheduler_factor'],
            lr_min=self.config['lr_min']
        )

        # Train model
        history = trainer.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            early_stopping_patience=self.config['early_stopping_patience']
        )

        # Save model
        trainer.save_model(DL_OUTPUT_FILES['trained_model'])

        # Save training history
        with open(DL_OUTPUT_FILES['training_history'], 'w') as f:
            json.dump(history, f, indent=2)

        self.execution_times['train_model'] = time.time() - step_start
        print(f"\n✓ Model training completed in {self.execution_times['train_model']:.2f} seconds")

        # Step 7: Evaluate on Test Set
        print("\n" + "="*70)
        print("STEP 7: EVALUATE ON TEST SET")
        print("="*70)
        step_start = time.time()

        test_metrics = trainer.evaluate(X_test, y_test, batch_size=self.config['batch_size'])

        # Save evaluation metrics
        metrics_to_save = {
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1_score': float(test_metrics['f1_score']),
            'roc_auc': float(test_metrics['roc_auc'])
        }

        with open(DL_OUTPUT_FILES['evaluation_metrics'], 'w') as f:
            json.dump(metrics_to_save, f, indent=2)

        self.results['test_metrics'] = test_metrics

        self.execution_times['evaluate'] = time.time() - step_start
        print(f"\n✓ Evaluation completed in {self.execution_times['evaluate']:.2f} seconds")

        # Step 8: Predict Full Raster
        print("\n" + "="*70)
        print("STEP 8: PREDICT FULL RASTER")
        print("="*70)
        step_start = time.time()

        predictor = RasterPredictor(
            model=model,
            device=self.config['device'],
            patch_size=self.config['patch_size'],
            batch_size=self.config['pred_batch_size']
        )

        classification_map, probability_map = predictor.predict_raster(
            feature_stack,
            valid_mask,
            stride=self.config['pred_stride'],
            normalize=True
        )

        # Save rasters (only 4-class multiclass map)
        predictor.save_rasters(
            data['metadata']['s2_before'],
            multiclass_path=DL_OUTPUT_FILES['multiclass_raster']  # Save 4-class raster only
        )

        self.execution_times['predict_raster'] = time.time() - step_start
        print(f"\n✓ Full raster prediction completed in {self.execution_times['predict_raster']:.2f} seconds")

        # Final Summary
        total_time = time.time() - total_start
        self._print_summary(test_metrics, total_time)

        return self.results

    def _print_config(self):
        """Print pipeline configuration"""
        print("\n" + "="*70)
        print("CNN PIPELINE CONFIGURATION")
        print("="*70)
        print(f"\nMODEL ARCHITECTURE:")
        print(f"  Model type: {self.config['model_type']}")
        print(f"  Patch size: {self.config['patch_size']}x{self.config['patch_size']}")
        print(f"  Input features: {self.config['n_features']}")
        print(f"  Output classes: {self.config['n_classes']}")
        print(f"  Dropout rate: {self.config['dropout_rate']}")

        print(f"\nTRAINING PARAMETERS:")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Weight decay: {self.config['weight_decay']}")
        print(f"  Early stopping patience: {self.config['early_stopping_patience']}")
        if self.config.get('use_lr_scheduler', False):
            print(f"\n  Learning Rate Scheduler:")
            print(f"    Type: {self.config.get('lr_scheduler_type', 'N/A')}")
            print(f"    Patience: {self.config.get('lr_scheduler_patience', 'N/A')}")
            print(f"    Factor: {self.config.get('lr_scheduler_factor', 'N/A')}")
            print(f"    Min LR: {self.config.get('lr_min', 'N/A')}")

        print(f"\nDATA SPLIT (Spatial-Aware):")
        print(f"  Cluster distance: {self.config['cluster_distance']}m")
        print(f"  Train: {self.config['train_size']*100:.0f}%")
        print(f"  Val: {self.config['val_size']*100:.0f}%")
        print(f"  Test: {self.config['test_size']*100:.0f}%")

        print(f"\nDEVICE: {self.config['device']}")
        print("="*70 + "\n")

    def _print_summary(self, test_metrics, total_time):
        """Print final pipeline summary"""
        print("\n" + "="*70)
        print("CNN PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)

        print("\n📊 MODEL PERFORMANCE (Test Set):")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:  {test_metrics['f1_score']:.4f} ({test_metrics['f1_score']*100:.2f}%)")
        print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f} ({test_metrics['roc_auc']*100:.2f}%)")

        print("\n⏱️  EXECUTION TIMES:")
        for step, duration in self.execution_times.items():
            print(f"  {step:20s}: {duration:7.2f}s ({duration/60:5.2f} min)")
        print(f"  {'TOTAL':20s}: {total_time:7.2f}s ({total_time/60:5.2f} min)")

        print("\n✅ All output files saved to: results/")
        print(f"  Model: {DL_OUTPUT_FILES['trained_model']}")
        print(f"  Classification map: {DL_OUTPUT_FILES['classification_raster']}")
        print(f"  Probability map: {DL_OUTPUT_FILES['probability_raster']}")
        print(f"  Metrics: {DL_OUTPUT_FILES['evaluation_metrics']}")
        print("="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='CNN Deforestation Detection Pipeline'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (overrides config)'
    )

    args = parser.parse_args()

    # Override config if arguments provided
    config_overrides = {}
    if args.epochs is not None:
        config_overrides['epochs'] = args.epochs
    if args.batch_size is not None:
        config_overrides['batch_size'] = args.batch_size
    if args.device is not None:
        config_overrides['device'] = args.device

    # Run pipeline
    pipeline = DeforestationCNNPipeline(config=config_overrides)
    pipeline.run()


if __name__ == '__main__':
    main()
