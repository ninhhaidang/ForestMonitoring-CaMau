"""
Main entry point for Random Forest Deforestation Detection Pipeline
Cà Mau Province - Sentinel-1 & Sentinel-2

Usage:
    python main.py                      # Run full pipeline
    python main.py --skip-vectorization # Skip vectorization step
"""

import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import common modules
from common.config import (
    print_config_summary, create_output_directories,
    verify_input_files, FEATURE_NAMES
)
from common.data_loader import DataLoader
from common.feature_extraction import FeatureExtraction
from common.evaluation import ModelEvaluator
from common.visualization import Visualizer

# Import Random Forest modules
from random_forest.train import RandomForestTrainer, TrainingDataExtractor
from random_forest.predict import RasterPredictor
from random_forest.vectorization import Vectorizer


class DeforestationDetectionPipeline:
    """
    Main pipeline orchestrator for Random Forest deforestation detection
    """

    def __init__(self, skip_vectorization=False):
        self.skip_vectorization = skip_vectorization
        self.execution_times = {}

    def run(self):
        """Execute the full pipeline"""
        total_start = time.time()

        print("\n" + "="*70)
        print("RANDOM FOREST DEFORESTATION DETECTION PIPELINE")
        print("="*70 + "\n")

        # Print configuration
        print_config_summary()
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

        # Step 4: Extract Training Data
        print("\n" + "="*70)
        print("STEP 4: EXTRACT TRAINING DATA")
        print("="*70)
        step_start = time.time()

        extractor = TrainingDataExtractor()
        training_df = extractor.extract_pixel_values(
            feature_stack, data['ground_truth'],
            data['metadata']['s2_before']['transform']
        )

        X = training_df[FEATURE_NAMES].values
        y = training_df['label'].values

        extractor.check_data_quality(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test = extractor.split_data(X, y)
        extractor.save_training_data()

        self.execution_times['extract_training'] = time.time() - step_start
        print(f"\n✓ Training data extraction completed in {self.execution_times['extract_training']:.2f} seconds")

        # Step 5: Train Random Forest
        print("\n" + "="*70)
        print("STEP 5: TRAIN RANDOM FOREST MODEL")
        print("="*70)
        step_start = time.time()

        trainer = RandomForestTrainer()
        model = trainer.train(X_train, y_train, X_val, y_val)
        trainer.save_model()

        self.execution_times['train_model'] = time.time() - step_start
        print(f"\n✓ Model training completed in {self.execution_times['train_model']:.2f} seconds")

        # Step 6: Model Evaluation
        print("\n" + "="*70)
        print("STEP 6: MODEL EVALUATION")
        print("="*70)
        step_start = time.time()

        evaluator = ModelEvaluator(model)
        val_metrics = evaluator.evaluate_validation(X_val, y_val)
        test_metrics = evaluator.evaluate_test(X_test, y_test)
        feature_importance_df = evaluator.calculate_feature_importance(FEATURE_NAMES)
        cv_scores = evaluator.cross_validate(X_train, y_train)
        evaluator.save_results()

        self.execution_times['evaluation'] = time.time() - step_start
        print(f"\n✓ Model evaluation completed in {self.execution_times['evaluation']:.2f} seconds")

        # Step 7: Predict Full Raster
        print("\n" + "="*70)
        print("STEP 7: PREDICT FULL RASTER")
        print("="*70)
        step_start = time.time()

        predictor = RasterPredictor(model)
        classification_map, probability_map = predictor.predict_raster(
            feature_stack, valid_mask, batch_size=10000
        )
        predictor.save_all_rasters(data['metadata']['s2_before'])

        self.execution_times['predict_raster'] = time.time() - step_start
        print(f"\n✓ Full raster prediction completed in {self.execution_times['predict_raster']:.2f} seconds")

        # Step 8: Vectorization (Optional)
        if not self.skip_vectorization:
            print("\n" + "="*70)
            print("STEP 8: VECTORIZATION")
            print("="*70)
            step_start = time.time()

            vectorizer = Vectorizer()
            polygons_gdf = vectorizer.vectorize_and_save(
                classification_map,
                data['metadata']['s2_before']['transform'],
                data['metadata']['s2_before']['crs'],
                apply_morphology=True,
                apply_simplification=True
            )

            self.execution_times['vectorization'] = time.time() - step_start
            print(f"\n✓ Vectorization completed in {self.execution_times['vectorization']:.2f} seconds")
        else:
            print("\n[SKIP] Vectorization skipped as requested")
            polygons_gdf = None

        # Step 9: Visualization
        print("\n" + "="*70)
        print("STEP 9: VISUALIZATION")
        print("="*70)
        step_start = time.time()

        visualizer = Visualizer()
        visualizer.create_all_visualizations(
            val_metrics=evaluator.val_metrics,
            test_metrics=evaluator.test_metrics,
            feature_importance_df=evaluator.feature_importance,
            cv_scores=evaluator.cv_scores,
            classification_map=classification_map,
            probability_map=probability_map,
            X_test=X_test,
            y_test=y_test,
            model=model,
            valid_mask=valid_mask
        )

        self.execution_times['visualization'] = time.time() - step_start
        print(f"\n✓ Visualization completed in {self.execution_times['visualization']:.2f} seconds")

        # Final Summary
        total_time = time.time() - total_start
        self.print_summary(test_metrics, total_time)

    def print_summary(self, test_metrics, total_time):
        """Print final pipeline summary"""
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
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
        print("="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Random Forest Deforestation Detection Pipeline'
    )
    parser.add_argument(
        '--skip-vectorization',
        action='store_true',
        help='Skip vectorization step (faster execution)'
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = DeforestationDetectionPipeline(
        skip_vectorization=args.skip_vectorization
    )
    pipeline.run()


if __name__ == '__main__':
    main()
