"""
Prediction functions for Hardware Trojan Detection
"""

import numpy as np
import pandas as pd
import joblib
import os

import config
from data_processing import feature_engineering

def load_model_and_scaler(model_path=config.MODEL_PATH, scaler_path=config.SCALER_PATH):
    """Load the trained model and scaler"""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {str(e)}")
        return None, None

def predict_trojan(new_data, model=None, scaler=None, threshold=0.5, 
                  model_path=config.MODEL_PATH, scaler_path=config.SCALER_PATH):
    """
    Make Trojan predictions on new data
    
    Parameters:
    -----------
    new_data : pandas.DataFrame
        The new data to make predictions on
    model : object, optional
        Pre-loaded model (if None, will load from disk)
    scaler : object, optional
        Pre-loaded scaler (if None, will load from disk)
    threshold : float, default=0.5
        Classification threshold for binary prediction
    model_path : str, default=config.MODEL_PATH
        Path to the saved model file
    scaler_path : str, default=config.SCALER_PATH
        Path to the saved scaler file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with original data and prediction results
    """
    # Load model and scaler if not provided
    if model is None or scaler is None:
        model, scaler = load_model_and_scaler(model_path, scaler_path)
        if model is None or scaler is None:
            return None
    
    try:
        # Apply feature engineering
        enhanced_data = feature_engineering(new_data)
        
        # Scale features
        X = enhanced_data.copy()
        if 'Label' in X.columns:
            X = X.drop('Label', axis=1)
            
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns
        )
        
        # Make predictions
        trojan_probs = model.predict_proba(X_scaled)[:, 1]
        predictions = (trojan_probs >= threshold).astype(int)
        
        # Add predictions to original data
        result_data = new_data.copy()
        result_data['Trojan_Probability'] = trojan_probs
        result_data['Is_Trojan'] = predictions
        
        # Calculate summary statistics
        trojan_count = predictions.sum()
        trojan_rate = trojan_count / len(predictions) if len(predictions) > 0 else 0
        
        print(f"Processed {len(new_data)} samples")
        print(f"Trojans detected: {trojan_count} ({trojan_rate:.2%})")
        
        return result_data
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def create_real_time_detector(model_path=config.MODEL_PATH, scaler_path=config.SCALER_PATH, threshold=0.5):
    """Create a real-time Trojan detection function"""
    # Load model and scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    if model is None or scaler is None:
        return None
    
    def detect_trojan(current, volt, power):
        """Real-time Trojan detection function"""
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'Current': [current],
            'Volt': [volt],
            'Power': [power]
        })
        
        # Apply feature engineering
        enhanced_data = feature_engineering(input_data)
        
        # Scale the features
        scaled_data = pd.DataFrame(
            scaler.transform(enhanced_data),
            columns=enhanced_data.columns
        )
        
        # Make prediction
        trojan_prob = model.predict_proba(scaled_data)[0, 1]
        is_trojan = trojan_prob >= threshold
        
        return {
            'is_trojan': bool(is_trojan),
            'probability': float(trojan_prob),
            'threshold': threshold
        }
    
    return detect_trojan

def batch_process_data(input_dir, output_dir=config.OUTPUT_DIR, 
                      model_path=config.MODEL_PATH, scaler_path=config.SCALER_PATH, 
                      threshold=0.5):
    """Process multiple CSV files in a directory"""
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    if model is None or scaler is None:
        return None
    
    results = []
    
    # Process all CSV files in the input directory
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            try:
                # Load data
                file_path = os.path.join(input_dir, file)
                data = pd.read_csv(file_path)
                print(f"Processing {file}: {len(data)} samples")
                
                # Make predictions
                result_data = predict_trojan(data, model, scaler, threshold)
                
                # Save results
                output_file = os.path.join(output_dir, f"results_{file}")
                result_data.to_csv(output_file, index=False)
                
                # Store summary
                trojans_detected = result_data['Is_Trojan'].sum()
                results.append({
                    'file': file,
                    'samples': len(data),
                    'trojans_detected': trojans_detected,
                    'detection_rate': trojans_detected / len(data) if len(data) > 0 else 0
                })
                
                print(f"Saved results to {output_file}")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                results.append({
                    'file': file,
                    'error': str(e)
                })
    
    # Create summary report
    summary_df = pd.DataFrame([r for r in results if 'error' not in r])
    if not summary_df.empty:
        total_samples = summary_df['samples'].sum()
        total_trojans = summary_df['trojans_detected'].sum()
        
        print("\n=== Batch Processing Summary ===")
        print(f"Total files processed: {len(results)}")
        print(f"Total samples: {total_samples}")
        print(f"Total trojans detected: {total_trojans}")
        print(f"Overall detection rate: {total_trojans/total_samples:.2%}")
        
        # Save summary report
        summary_file = os.path.join(output_dir, "batch_processing_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary to {summary_file}")
    
    return results

def analyze_trojan_characteristics(data_with_predictions, output_dir=config.OUTPUT_DIR):
    """Analyze characteristics of detected Trojans"""
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if 'Is_Trojan' not in data_with_predictions.columns:
        print("Error: DataFrame missing 'Is_Trojan' column")
        return None
    
    # Separate Trojan and non-Trojan samples
    trojan_samples = data_with_predictions[data_with_predictions['Is_Trojan'] == 1]
    non_trojan_samples = data_with_predictions[data_with_predictions['Is_Trojan'] == 0]
    
    # Calculate statistics
    print("\n=== Trojan Characteristics Analysis ===")
    print(f"Total samples: {len(data_with_predictions)}")
    print(f"Detected Trojans: {len(trojan_samples)} ({len(trojan_samples)/len(data_with_predictions)*100:.2f}%)")
    
    # Feature comparison between Trojan and non-Trojan
    feature_comparison = {}
    basic_features = ['Current', 'Volt', 'Power']
    
    for feature in basic_features:
        if feature in data_with_predictions.columns:
            trojan_mean = trojan_samples[feature].mean()
            non_trojan_mean = non_trojan_samples[feature].mean()
            difference_pct = ((trojan_mean - non_trojan_mean) / non_trojan_mean * 100) if non_trojan_mean != 0 else 0
            
            feature_comparison[feature] = {
                'Trojan_Mean': trojan_mean,
                'Non_Trojan_Mean': non_trojan_mean,
                'Difference_Percent': difference_pct
            }
    
    # Print feature comparison
    print("\nFeature Comparison (Trojan vs Non-Trojan):")
    for feature, stats in feature_comparison.items():
        print(f"  {feature}:")
        print(f"    - Trojan Mean: {stats['Trojan_Mean']:.4f}")
        print(f"    - Non-Trojan Mean: {stats['Non_Trojan_Mean']:.4f}")
        print(f"    - Difference: {stats['Difference_Percent']:.2f}%")
    
    # Visualize trojan vs non-trojan characteristic differences
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(basic_features):
        if feature in data_with_predictions.columns:
            plt.subplot(2, 2, i+1)
            sns.kdeplot(data=trojan_samples, x=feature, label="Trojan", color="red")
            sns.kdeplot(data=non_trojan_samples, x=feature, label="Non-Trojan", color="blue")
            plt.title(f"{feature} Distribution: Trojan vs Non-Trojan")
            plt.legend()
    
    # Correlation between probability and features
    plt.subplot(2, 2, 4)
    correlation_data = []
    
    for feature in basic_features:
        if feature in data_with_predictions.columns:
            corr = data_with_predictions[feature].corr(data_with_predictions['Trojan_Probability'])
            correlation_data.append({'Feature': feature, 'Correlation': corr})
    
    corr_df = pd.DataFrame(correlation_data)
    sns.barplot(x='Feature', y='Correlation', data=corr_df)
    plt.title("Correlation with Trojan Probability")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trojan_characteristics.png"))
    plt.close()
    
    return feature_comparison