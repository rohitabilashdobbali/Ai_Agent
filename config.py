"""
Configuration settings for the Hardware Trojan Detection System
"""

import os

# API Key for External Services
LLAMA_API_KEY = "API_KEY"
os.environ["LLAMA_API_KEY"] = LLAMA_API_KEY

# Paths
DATA_PATH = "A.csv"                    # Path to your labeled dataset
MODEL_PATH = "ensemble_model.pkl"      # Where to save the trained ensemble
SCALER_PATH = "scaler.pkl"             # Where to save the fitted scaler
FEATURE_IMP_PATH = "feature_importance.png"  # Feature importance visualization
OUTPUT_DIR = "./results"               # Directory for storing results
OUTPUT_DIR = "./results"
# Model Training Parameters
TEST_SIZE = 0.2                        # Proportion of data for test set
VALIDATION_SIZE = 0.15                 # Proportion of data for validation set
RANDOM_STATE = 42                      # Random seed for reproducibility
N_ITER_SEARCH = 100                    # Number of iterations for hyperparameter search
N_CV_FOLDS = 5                         # Number of cross-validation folds

# Model Thresholds
DEFAULT_THRESHOLD = 0.5                # Default classification threshold
