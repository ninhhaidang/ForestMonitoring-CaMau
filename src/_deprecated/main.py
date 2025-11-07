"""
MAIN PIPELINE
Complete Random Forest Deforestation Detection Pipeline
Cà Mau Province - Sentinel-1 & Sentinel-2

This script runs the complete 9-step pipeline:
1. Setup & Load Data
2. Feature Engineering
3. Extract Training Data
4. Train Random Forest
5. Model Evaluation
6. Predict Full Raster
7. Vectorization
8. Visualization
"""

import sys
import logging
import time
from pathlib import Path
import argparse

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

# Import all modules
from config import (
    print_config_summary, create_output_directories,
    verify_input_files, FEATURE_NAMES, LOG_CONFIG
)
from step1_2_setup_and_load_data import DataLoader
from step3_feature_engineering import FeatureEngineering
from step4_extract_training_data import TrainingDataExtractor
from step5_train_random_forest import RandomForestTrainer
from step6_model_evaluation import ModelEvaluator
from step7_predict_full_raster import RasterPredictor
from step8_vectorization import Vectorizer
from step9_visualization import Visualizer

# Setup logging
logging.basicConfig(**LOG_CONFIG)
logger = logging.getLogger(__name__)


class DeforestationDetectionPipeline:
    """Complete pipeline for deforestation detection"""

    def __init__(self):
        """Initialize pipeline"""
        self.loader = None
        self.data = None
        self.engineer = None
        self.feature_stack = None
        self.valid_mask = None
        self.extractor = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.trainer = None
        self.model = None
        self.evaluator = None
        self.predictor = None
        self.classification_map = None
        self.probability_map = None
        self.vectorizer = None
        self.polygons_gdf = None
        self.visualizer = None

        self.execution_times = {}
        self.total_start_time = None

    def run_step1_2(self):
        """Step 1 & 2: Setup and Load Data"""
        logger.info("\n" + "="*70)
        logger.info("EXECUTING STEP 1 & 2: SETUP AND LOAD DATA")
        logger.info("="*70)

        start_time = time.time()

        # Load all data
        self.loader = DataLoader()
        self.data = self.loader.load_all()

        self.execution_times['step1_2'] = time.time() - start_time
        logger.info(f"\n[OK] Step 1 & 2 completed in {self.execution_times['step1_2']:.2f} seconds")

    def run_step3(self):
        """Step 3: Feature Engineering"""
        logger.info("\n" + "="*70)
        logger.info("EXECUTING STEP 3: FEATURE ENGINEERING")
        logger.info("="*70)

        start_time = time.time()

        # Engineer features
        self.engineer = FeatureEngineering()
        self.feature_stack, self.valid_mask = self.engineer.engineer_features(
            self.data['s2_before'],
            self.data['s2_after'],
            self.data['s1_before'],
            self.data['s1_after']
        )

        self.execution_times['step3'] = time.time() - start_time
        logger.info(f"\n[OK] Step 3 completed in {self.execution_times['step3']:.2f} seconds")

    def run_step4(self):
        """Step 4: Extract Training Data"""
        logger.info("\n" + "="*70)
        logger.info("EXECUTING STEP 4: EXTRACT TRAINING DATA")
        logger.info("="*70)

        start_time = time.time()

        # Extract training data
        self.extractor = TrainingDataExtractor()
        training_df = self.extractor.extract_pixel_values(
            self.feature_stack,
            self.data['ground_truth'],
            self.data['metadata']['s2_before']['transform']
        )

        # Get features and labels
        X = training_df[FEATURE_NAMES].values
        y = training_df['label'].values

        # Check data quality
        self.extractor.check_data_quality(X, y)

        # Split data
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            self.extractor.split_data(X, y)

        # Save training data
        self.extractor.save_training_data()

        self.execution_times['step4'] = time.time() - start_time
        logger.info(f"\n[OK] Step 4 completed in {self.execution_times['step4']:.2f} seconds")

    def run_step5(self):
        """Step 5: Train Random Forest"""
        logger.info("\n" + "="*70)
        logger.info("EXECUTING STEP 5: TRAIN RANDOM FOREST")
        logger.info("="*70)

        start_time = time.time()

        # Train model
        self.trainer = RandomForestTrainer()
        self.model = self.trainer.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val
        )

        # Save model
        self.trainer.save_model()

        self.execution_times['step5'] = time.time() - start_time
        logger.info(f"\n[OK] Step 5 completed in {self.execution_times['step5']:.2f} seconds")

    def run_step6(self):
        """Step 6: Model Evaluation"""
        logger.info("\n" + "="*70)
        logger.info("EXECUTING STEP 6: MODEL EVALUATION")
        logger.info("="*70)

        start_time = time.time()

        # Evaluate model
        self.evaluator = ModelEvaluator(self.model)
        self.evaluator.evaluate_all(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            self.X_test, self.y_test,
            FEATURE_NAMES
        )

        # Save results
        self.evaluator.save_results()

        self.execution_times['step6'] = time.time() - start_time
        logger.info(f"\n[OK] Step 6 completed in {self.execution_times['step6']:.2f} seconds")

    def run_step7(self):
        """Step 7: Predict Full Raster"""
        logger.info("\n" + "="*70)
        logger.info("EXECUTING STEP 7: PREDICT FULL RASTER")
        logger.info("="*70)

        start_time = time.time()

        # Predict on full raster
        self.predictor = RasterPredictor(self.model)
        self.classification_map, self.probability_map = self.predictor.predict_raster(
            self.feature_stack,
            self.valid_mask,
            batch_size=10000
        )

        # Save rasters
        self.predictor.save_all_rasters(self.data['metadata']['s2_before'])

        self.execution_times['step7'] = time.time() - start_time
        logger.info(f"\n[OK] Step 7 completed in {self.execution_times['step7']:.2f} seconds")

    def run_step8(self, apply_vectorization: bool = True):
        """Step 8: Vectorization (Optional)"""
        if not apply_vectorization:
            logger.info("\n[SKIP] Step 8: Vectorization skipped (optional)")
            return

        logger.info("\n" + "="*70)
        logger.info("EXECUTING STEP 8: VECTORIZATION")
        logger.info("="*70)

        start_time = time.time()

        # Vectorize
        self.vectorizer = Vectorizer()
        self.polygons_gdf = self.vectorizer.vectorize_and_save(
            self.classification_map,
            self.data['metadata']['s2_before']['transform'],
            self.data['metadata']['s2_before']['crs'],
            apply_morphology=True,
            apply_simplification=True
        )

        self.execution_times['step8'] = time.time() - start_time
        logger.info(f"\n[OK] Step 8 completed in {self.execution_times['step8']:.2f} seconds")

    def run_step9(self):
        """Step 9: Visualization"""
        logger.info("\n" + "="*70)
        logger.info("EXECUTING STEP 9: VISUALIZATION")
        logger.info("="*70)

        start_time = time.time()

        # Create visualizations
        self.visualizer = Visualizer()
        self.visualizer.create_all_visualizations(
            val_metrics=self.evaluator.val_metrics,
            test_metrics=self.evaluator.test_metrics,
            feature_importance_df=self.evaluator.feature_importance,
            cv_scores=self.evaluator.cv_scores,
            classification_map=self.classification_map,
            probability_map=self.probability_map,
            X_test=self.X_test,
            y_test=self.y_test,
            model=self.model,
            valid_mask=self.valid_mask
        )

        self.execution_times['step9'] = time.time() - start_time
        logger.info(f"\n[OK] Step 9 completed in {self.execution_times['step9']:.2f} seconds")

    def run_complete_pipeline(self, skip_vectorization: bool = False):
        """
        Run complete pipeline

        Args:
            skip_vectorization: Skip vectorization step (optional)
        """
        self.total_start_time = time.time()

        logger.info("\n" + "="*70)
        logger.info("RANDOM FOREST DEFORESTATION DETECTION PIPELINE")
        logger.info("Ca Mau Province - Sentinel-1 & Sentinel-2")
        logger.info("="*70)

        # Print configuration
        print_config_summary()

        # Create output directories
        create_output_directories()

        # Verify input files
        if not verify_input_files():
            logger.error("Please ensure all required input files exist")
            return False

        try:
            # Run all steps
            self.run_step1_2()  # Load data
            self.run_step3()    # Feature engineering
            self.run_step4()    # Extract training data
            self.run_step5()    # Train model
            self.run_step6()    # Evaluate model
            self.run_step7()    # Predict full raster
            self.run_step8(apply_vectorization=not skip_vectorization)  # Vectorization
            self.run_step9()    # Visualization

            # Print final summary
            self.print_final_summary()

            return True

        except Exception as e:
            logger.error(f"\n✗ Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def print_final_summary(self):
        """Print final pipeline summary"""
        total_time = time.time() - self.total_start_time

        logger.info("\n" + "="*70)
        logger.info("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)

        logger.info("\nExecution Times:")
        logger.info(f"  Step 1-2 (Load Data):         {self.execution_times.get('step1_2', 0):8.2f} seconds")
        logger.info(f"  Step 3   (Feature Engineering): {self.execution_times.get('step3', 0):8.2f} seconds")
        logger.info(f"  Step 4   (Extract Training):    {self.execution_times.get('step4', 0):8.2f} seconds")
        logger.info(f"  Step 5   (Train Model):         {self.execution_times.get('step5', 0):8.2f} seconds")
        logger.info(f"  Step 6   (Evaluation):          {self.execution_times.get('step6', 0):8.2f} seconds")
        logger.info(f"  Step 7   (Predict Raster):      {self.execution_times.get('step7', 0):8.2f} seconds")
        if 'step8' in self.execution_times:
            logger.info(f"  Step 8   (Vectorization):       {self.execution_times.get('step8', 0):8.2f} seconds")
        logger.info(f"  Step 9   (Visualization):       {self.execution_times.get('step9', 0):8.2f} seconds")
        logger.info(f"  " + "-"*50)
        logger.info(f"  TOTAL TIME:                     {total_time:8.2f} seconds ({total_time/60:.2f} minutes)")

        logger.info("\n📊 Model Performance:")
        if self.evaluator and self.evaluator.test_metrics:
            test = self.evaluator.test_metrics
            logger.info(f"  Test Accuracy:  {test['accuracy']:.4f} ({test['accuracy']*100:.2f}%)")
            logger.info(f"  Test Precision: {test['precision']:.4f} ({test['precision']*100:.2f}%)")
            logger.info(f"  Test Recall:    {test['recall']:.4f} ({test['recall']*100:.2f}%)")
            logger.info(f"  Test F1-Score:  {test['f1_score']:.4f} ({test['f1_score']*100:.2f}%)")
            logger.info(f"  Test ROC-AUC:   {test['roc_auc']:.4f} ({test['roc_auc']*100:.2f}%)")

        logger.info("\n📁 Output Files:")
        from config import OUTPUT_FILES, RESULTS_DIR
        logger.info(f"  Results directory: {RESULTS_DIR}")
        logger.info(f"  - Classification raster: {OUTPUT_FILES['classification_raster'].name}")
        logger.info(f"  - Probability raster: {OUTPUT_FILES['probability_raster'].name}")
        logger.info(f"  - Trained model: {OUTPUT_FILES['trained_model'].name}")
        logger.info(f"  - Feature importance: {OUTPUT_FILES['feature_importance'].name}")
        logger.info(f"  - Evaluation metrics: {OUTPUT_FILES['evaluation_metrics'].name}")
        if self.polygons_gdf is not None and len(self.polygons_gdf) > 0:
            logger.info(f"  - Deforestation polygons: {OUTPUT_FILES['polygons_geojson'].name}")

        logger.info("\n" + "="*70)
        logger.info("🎉 ALL DONE! 🎉")
        logger.info("="*70 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Random Forest Deforestation Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py

  # Run without vectorization (faster)
  python main.py --skip-vectorization

  # Run specific steps only
  python main.py --steps 1 2 3 4 5
        """
    )

    parser.add_argument(
        '--skip-vectorization',
        action='store_true',
        help='Skip vectorization step (step 8)'
    )

    parser.add_argument(
        '--steps',
        nargs='+',
        type=int,
        choices=range(1, 10),
        help='Run specific steps only (e.g., --steps 1 2 3)'
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = DeforestationDetectionPipeline()

    if args.steps:
        logger.info(f"Running specific steps: {args.steps}")
        # TODO: Implement selective step execution
        logger.warning("Selective step execution not yet implemented. Running full pipeline.")

    success = pipeline.run_complete_pipeline(
        skip_vectorization=args.skip_vectorization
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
