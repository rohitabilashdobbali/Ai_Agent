"""
Main script for Hardware Trojan Detection System
"""

import time
import os
from datetime import datetime

import config
from data_processing import (
    load_data, feature_engineering, visualize_features, preprocess_data, 
    print_dataset_stats, split_data, handle_class_imbalance
)
from model_training import (
    train_models, build_ensemble_model, save_models, plot_feature_importance
)
from model_evaluation import (
    find_optimal_threshold, evaluate_model, compare_models, extract_top_features,
    run_cross_validation, generate_report
)
from prediction import predict_trojan, create_real_time_detector, analyze_trojan_characteristics

def run_pipeline(data_path=config.DATA_PATH, output_dir=config.OUTPUT_DIR):
    """Run the complete hardware Trojan detection pipeline"""
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*80)
    print(f"HARDWARE TROJAN DETECTION PIPELINE - {timestamp}")
    print("="*80)

    # Initialize result log
    log_file = os.path.join(output_dir, f"trojan_detection_log_{timestamp}.txt")

    def log(message):
        """Log message to console and file"""
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")

    # Step 1: Data Loading
    log("\nSTEP 1: Data Loading")
    try:
        data = load_data(data_path)
        log(f"Successfully loaded {len(data)} samples from {data_path}")
    except Exception as e:
        log(f"ERROR: Failed to load data - {str(e)}")
        return {"success": False, "error": str(e)}

    # Step 2: Feature Engineering
    log("\nSTEP 2: Feature Engineering")
    try:
        enhanced_data = feature_engineering(data)
        log(f"Enhanced data shape: {enhanced_data.shape}")
        log(f"Created {enhanced_data.shape[1] - data.shape[1]} new features")
    except Exception as e:
        log(f"ERROR: Feature engineering failed - {str(e)}")
        return {"success": False, "error": str(e)}

    # Step 3: Data Visualization
    log("\nSTEP 3: Data Visualization")
    try:
        visualize_features(enhanced_data, output_dir)
        log("Created feature visualizations")
    except Exception as e:
        log(f"WARNING: Visualization failed - {str(e)}")
        # Continue even if visualization fails

    # Step 4: Data Preprocessing
    log("\nSTEP 4: Data Preprocessing")
    try:
        X, y, scaler = preprocess_data(enhanced_data)
        print_dataset_stats(y, output_dir)
        log(f"Preprocessed data shape: {X.shape}")
    except Exception as e:
        log(f"ERROR: Preprocessing failed - {str(e)}")
        return {"success": False, "error": str(e)}

    # Step 5: Data Splitting
    log("\nSTEP 5: Data Splitting")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    except Exception as e:
        log(f"ERROR: Data splitting failed - {str(e)}")
        return {"success": False, "error": str(e)}

    # Step 6: Handle Class Imbalance
    log("\nSTEP 6: Handling Class Imbalance")
    try:
        X_train_res, y_train_res = handle_class_imbalance(X_train, y_train)
    except Exception as e:
        log(f"WARNING: SMOTE resampling failed - {str(e)}")
        log("Continuing with original imbalanced data")
        X_train_res, y_train_res = X_train, y_train

    # Step 7: Model Training
    log("\nSTEP 7: Model Training")
    try:
        models, svm_included = train_models(X_train_res, y_train_res)
        
        # Save feature importance plots
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                plot_feature_importance(
                    model, 
                    X.columns, 
                    f"{name.upper()} Feature Importance", 
                    os.path.join(output_dir, f"{name}_feature_importance.png")
                )
    except Exception as e:
        log(f"ERROR: Model training failed - {str(e)}")
        return {"success": False, "error": str(e)}

    # Step 8: Build Ensemble Model
    log("\nSTEP 8: Building Ensemble Model")
    try:
        ensemble_model = build_ensemble_model(models)
        ensemble_model.fit(X_train_res, y_train_res)
        log("Ensemble model trained successfully")
        
        # Save models and scaler
        save_info = save_models(models, ensemble_model, scaler, output_dir)
    except Exception as e:
        log(f"ERROR: Ensemble building failed - {str(e)}")
        return {"success": False, "error": str(e)}

    # Step 9: Find Optimal Threshold
    log("\nSTEP 9: Finding Optimal Threshold")
    try:
        optimal_threshold = find_optimal_threshold(ensemble_model, X_val, y_val, output_dir)
        log(f"Optimal threshold: {optimal_threshold:.3f}")
    except Exception as e:
        log(f"WARNING: Threshold optimization failed - {str(e)}")
        optimal_threshold = 0.5
        log(f"Using default threshold: {optimal_threshold}")

    # Step 10: Model Evaluation
    log("\nSTEP 10: Model Evaluation")
    metrics = {}
    try:
        # Individual models evaluation
        for name, model in models.items():
            log(f"Evaluating {name.upper()} model...")
            metrics[name] = evaluate_model(
                model, X_test, y_test, optimal_threshold, name.upper(), output_dir
            )
        
        # Ensemble model evaluation
        log("Evaluating ENSEMBLE model...")
        metrics['ensemble'] = evaluate_model(
            ensemble_model, X_test, y_test, optimal_threshold, "Ensemble Model", output_dir
        )
    except Exception as e:
        log(f"ERROR: Model evaluation failed - {str(e)}")
        return {"success": False, "error": str(e)}

    # Step 11: Model Comparison
    log("\nSTEP 11: Model Comparison")
    try:
        comparison_df = compare_models(metrics, output_dir)
    except Exception as e:
        log(f"WARNING: Model comparison failed - {str(e)}")

    # Step 12: Feature Importance Analysis
    log("\nSTEP 12: Feature Importance Analysis")
    try:
        top_features = extract_top_features(models, X.columns, output_dir)
    except Exception as e:
        log(f"WARNING: Feature importance analysis failed - {str(e)}")
        top_features = []

    # Step 13: Cross-Validation (optional, can be time-consuming)
    log("\nSTEP 13: Cross-Validation")
    try:
        cv_results = run_cross_validation(enhanced_data, X.columns, n_folds=config.N_CV_FOLDS, output_dir=output_dir)
        log("Cross-validation completed successfully")
    except Exception as e:
        log(f"WARNING: Cross-validation failed - {str(e)}")

    # Step 14: Analyze detected Trojans
    log("\nSTEP 14: Trojan Characteristics Analysis")
    try:
        # Make predictions on test data
        test_preds = ensemble_model.predict_proba(X_test)[:, 1]
        test_labels = (test_preds >= optimal_threshold).astype(int)
        
        # Reconstruct test data with original features and predictions
        test_data_with_preds = data.iloc[y_test.index].copy() if hasattr(y_test, 'index') else None
        
        if test_data_with_preds is not None:
            test_data_with_preds['Trojan_Probability'] = test_preds
            test_data_with_preds['Is_Trojan'] = test_labels
            
            # Analyze trojan characteristics
            trojan_analysis = analyze_trojan_characteristics(test_data_with_preds, output_dir)
            log("Trojan characteristics analysis completed")
        else:
            log("WARNING: Could not reconstruct test data for trojan analysis")
    except Exception as e:
        log(f"WARNING: Trojan characteristics analysis failed - {str(e)}")

    # Step 15: Generate Report
    log("\nSTEP 15: Generating Final Report")
    try:
        end_time = time.time()
        execution_time = (end_time - start_time) / 60  # minutes
        
        report = generate_report(
            comparison_df,
            top_features,
            optimal_threshold,
            execution_time,
            output_dir
        )
        
        log(f"Saved final report to {os.path.join(output_dir, 'trojan_detection_report.txt')}")
    except Exception as e:
        log(f"WARNING: Report generation failed - {str(e)}")

    # Final summary
    log("\n" + "="*80)
    log("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    log(f"Total execution time: {(time.time() - start_time)/60:.2f} minutes")
    best_model = comparison_df.iloc[comparison_df['F1 Score'].idxmax()]['Model'] if 'comparison_df' in locals() else "Unknown"
    log(f"Best model: {best_model}")
    log(f"All outputs saved to: {output_dir}")
    log("="*80)
    
    return {
        "success": True,
        "execution_time": (time.time() - start_time)/60,
        "best_model": best_model,
        "optimal_threshold": optimal_threshold,
        "output_dir": output_dir
    }

def quick_predict(data_path, model_path=config.MODEL_PATH, scaler_path=config.SCALER_PATH, threshold=None):
    """Quick prediction function for new data"""
    # Load data
    data = load_data(data_path)
    
    # If threshold not provided, use default
    if threshold is None:
        threshold = config.DEFAULT_THRESHOLD
    
    # Make predictions
    results = predict_trojan(data, model_path=model_path, scaler_path=scaler_path, threshold=threshold)
    
    return results

def create_detector():
    """Create a real-time detector function"""
    detector = create_real_time_detector()
    if detector:
        print("Real-time detector created successfully")
        print("Example usage:")
        print("  result = detector(current=10.5, volt=5.2, power=50.1)")
        print("  print(result['is_trojan'])")
    else:
        print("Failed to create real-time detector")
    
    return detector

if __name__ == "__main__":
    # Run the complete pipeline
    result = run_pipeline()
    
    if result["success"]:
        print("\nPipeline completed successfully!")
        print(f"Best model: {result['best_model']}")
        print(f"Optimal threshold: {result['optimal_threshold']:.3f}")
        print(f"Execution time: {result['execution_time']:.2f} minutes")
    else:
        print("\nPipeline failed:")
        print(result["error"])