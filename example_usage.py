"""
Example usage of the Hardware Trojan Detection System
"""

import pandas as pd
import os
import joblib

# Import modules from the trojan detection system
import config
from data_processing import load_data, feature_engineering
from prediction import predict_trojan, create_real_time_detector, analyze_trojan_characteristics
from main import run_pipeline, quick_predict, create_detector

def example_train_and_evaluate():
    """Example: Train and evaluate the trojan detection model"""
    print("\n=== Example: Training and Evaluating the Model ===")
    
    # Run the complete pipeline
    result = run_pipeline(data_path="A.csv", output_dir="./results")
    
    if result["success"]:
        print("\nSuccess! Model trained and evaluated.")
        print(f"Best model: {result['best_model']}")
        print(f"Optimal threshold: {result['optimal_threshold']:.3f}")
        print(f"Execution time: {result['execution_time']:.2f} minutes")
        print(f"All results saved to: {result['output_dir']}")
    else:
        print(f"\nFailed to train model: {result['error']}")

def example_predict_new_data():
    """Example: Making predictions on new data"""
    print("\n=== Example: Making Predictions on New Data ===")
    
    # Check if the model and scaler exist
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.SCALER_PATH):
        print("Model or scaler not found. Please train the model first.")
        return
    
    # Option 1: Using the quick_predict function
    print("\nOption 1: Using quick_predict function")
    new_data_path = "new_samples.csv"  # Path to your new data
    
    # Create example data if file doesn't exist
    if not os.path.exists(new_data_path):
        print(f"Creating example data file: {new_data_path}")
        example_data = pd.DataFrame({
            'Current': [10.2, 11.5, 9.8, 12.3, 10.0],
            'Volt': [5.0, 5.1, 4.9, 5.3, 6.2],
            'Power': [51.0, 58.6, 48.0, 65.2, 62.0]
        })
        example_data.to_csv(new_data_path, index=False)
    
    # Make predictions
    results = quick_predict(new_data_path)
    
    if results is not None:
        print("\nPrediction Results:")
        print(results[['Current', 'Volt', 'Power', 'Trojan_Probability', 'Is_Trojan']].head())
        
        # Save results
        results.to_csv("prediction_results.csv", index=False)
        print("Results saved to 'prediction_results.csv'")
        
        # Analyze characteristics
        print("\nAnalyzing trojan characteristics...")
        analyze_trojan_characteristics(results)
    
    # Option 2: Using the predict_trojan function directly
    print("\nOption 2: Using predict_trojan function directly")
    
    # Load model and scaler
    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    
    # Create example data
    new_data = pd.DataFrame({
        'Current': [9.9, 12.0, 11.3],
        'Volt': [5.0, 5.5, 4.8],
        'Power': [49.5, 66.0, 54.2]
    })
    
    # Make predictions
    results2 = predict_trojan(new_data, model, scaler)
    
    if results2 is not None:
        print("\nDirect Prediction Results:")
        print(results2[['Current', 'Volt', 'Power', 'Trojan_Probability', 'Is_Trojan']])
        
def example_real_time_detection():
    """Example: Real-time trojan detection"""
    print("\n=== Example: Real-time Trojan Detection ===")
    
    # Create detector function
    detector = create_detector()
    
    if detector is None:
        print("Failed to create detector. Please train the model first.")
        return
    
    # Example usage of real-time detector
    test_cases = [
        # Normal case
        {'current': 10.1, 'volt': 5.0, 'power': 50.5},
        # Possible trojan case
        {'current': 12.5, 'volt': 5.2, 'power': 65.0},
        # Another test case
        {'current': 9.8, 'volt': 4.9, 'power': 48.0}
    ]
    
    print("\nTesting real-time detector with example circuits:")
    for i, case in enumerate(test_cases):
        result = detector(case['current'], case['volt'], case['power'])
        print(f"\nCircuit {i+1}:")
        print(f"  Current: {case['current']} A")
        print(f"  Voltage: {case['volt']} V")
        print(f"  Power: {case['power']} W")
        print(f"  Trojan Detection: {'POSITIVE' if result['is_trojan'] else 'NEGATIVE'}")
        print(f"  Confidence: {abs(result['probability'] - 0.5) * 2:.2%}")
        print(f"  Probability: {result['probability']:.4f}")

def run_examples():
    """Run all examples"""
    print("HARDWARE TROJAN DETECTION SYSTEM - EXAMPLES")
    print("===========================================")
    
    while True:
        print("\nAvailable examples:")
        print("1. Train and evaluate model")
        print("2. Make predictions on new data")
        print("3. Demonstrate real-time detection")
        print("4. Run all examples")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-4): ")
        
        if choice == '1':
            example_train_and_evaluate()
        elif choice == '2':
            example_predict_new_data()
        elif choice == '3':
            example_real_time_detection()
        elif choice == '4':
            example_train_and_evaluate()
            example_predict_new_data()
            example_real_time_detection()
        elif choice == '0':
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    run_examples()