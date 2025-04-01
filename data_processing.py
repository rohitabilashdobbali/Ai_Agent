"""
Data loading, preprocessing, and feature engineering for Hardware Trojan Detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

import config

def load_data(file_path):
    """Load data with error handling"""
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded dataset from {file_path}")
        print(f"Dataset shape: {data.shape}")
        print("Columns in dataset:", data.columns.tolist())
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def feature_engineering(data):
    """Enhanced feature engineering with more sophisticated features"""
    # Create a copy to avoid modifying the original dataframe
    data_copy = data.copy()
    
    # Basic ratio features
    data_copy['Power_per_Current'] = data_copy.apply(lambda row: row['Power'] / row['Current'] if row['Current'] != 0 else 0, axis=1)
    data_copy['Power_per_Volt'] = data_copy.apply(lambda row: row['Power'] / row['Volt'] if row['Volt'] != 0 else 0, axis=1)
    data_copy['Volt_per_Current'] = data_copy.apply(lambda row: row['Volt'] / row['Current'] if row['Current'] != 0 else 0, axis=1)
    
    # Polynomial features for capturing non-linear relationships
    data_copy['Power_squared'] = data_copy['Power'] ** 2
    data_copy['Current_squared'] = data_copy['Current'] ** 2
    data_copy['Volt_squared'] = data_copy['Volt'] ** 2
    
    # Interaction features
    data_copy['Power_x_Current'] = data_copy['Power'] * data_copy['Current']
    data_copy['Power_x_Volt'] = data_copy['Power'] * data_copy['Volt']
    data_copy['Current_x_Volt'] = data_copy['Current'] * data_copy['Volt']
    
    # Log-transformed features to handle skewness
    data_copy['Power_log'] = np.log1p(np.abs(data_copy['Power']))
    data_copy['Current_log'] = np.log1p(np.abs(data_copy['Current']))
    data_copy['Volt_log'] = np.log1p(np.abs(data_copy['Volt']))
    
    # Statistical features
    data_copy['Power_zscore'] = (data_copy['Power'] - data_copy['Power'].mean()) / data_copy['Power'].std()
    data_copy['Current_zscore'] = (data_copy['Current'] - data_copy['Current'].mean()) / data_copy['Current'].std()
    data_copy['Volt_zscore'] = (data_copy['Volt'] - data_copy['Volt'].mean()) / data_copy['Volt'].std()
    
    # Replace infinities and NaNs with zeros
    data_copy = data_copy.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Print feature engineering summary
    print(f"Original features: {list(data.columns)}")
    print(f"Enhanced features: {list(data_copy.columns)}")
    print(f"Added {len(data_copy.columns) - len(data.columns)} new features")
    
    return data_copy

def detect_outliers(data, features, threshold=3):
    """Detect outliers using z-score method"""
    outliers = {}
    for feature in features:
        if feature in data.columns and pd.api.types.is_numeric_dtype(data[feature]):
            z_scores = np.abs((data[feature] - data[feature].mean()) / data[feature].std())
            outliers[feature] = data[z_scores > threshold].index.tolist()
    
    # Find common outliers across multiple features
    if outliers:
        all_outliers = set().union(*[set(ids) for ids in outliers.values()])
        print(f"Detected {len(all_outliers)} potential outliers across all features")
    
    return outliers

def visualize_features(data, output_dir=config.OUTPUT_DIR):
    """Create comprehensive feature visualizations"""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if "Label" not in data.columns:
        print("No 'Label' column; skipping visualizations.")
        return
    
    # 1. Create correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    corr = data[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=False, center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()
    
    # 2. Distribution plots for key features
    basic_features = ["Current", "Volt", "Power"]
    for feature in basic_features:
        plt.figure(figsize=(15, 5))
        
        # Distribution plot
        plt.subplot(1, 3, 1)
        sns.histplot(data=data, x=feature, hue="Label", kde=True, 
                    element="step", palette="Set1", common_norm=False)
        plt.title(f"{feature} Distribution by Class")
        
        # Box plot
        plt.subplot(1, 3, 2)
        sns.boxplot(x="Label", y=feature, data=data, palette="Set1")
        plt.title(f"{feature} Box Plot by Class")
        
        # Violin plot
        plt.subplot(1, 3, 3)
        sns.violinplot(x="Label", y=feature, data=data, palette="Set1", inner="quartile")
        plt.title(f"{feature} Violin Plot by Class")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{feature}_analysis.png"))
        plt.close()
    
    # 3. Pairplot for main features
    plt.figure(figsize=(12, 10))
    sns.pairplot(data[basic_features + ["Label"]], hue="Label", palette="Set1", diag_kind="kde")
    plt.suptitle("Feature Pair Relationships", y=1.02)
    plt.savefig(os.path.join(output_dir, "feature_pairplot.png"))
    plt.close()
    
    print(f"Saved visualizations to {output_dir}")

def preprocess_data(data):
    """Advanced preprocessing pipeline with robust scaling"""
    target = "Label"
    if target not in data.columns:
        raise KeyError(f"Target '{target}' not found. Available columns: {data.columns.tolist()}")
    
    print("\n=== Data Preprocessing ===")
    
    # Separate features and target first
    X = data.drop(target, axis=1)
    y = data[target]
    
    # Handle categorical features if any (excluding the Label column since we already separated it)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"One-hot encoding {len(categorical_cols)} categorical features")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Apply power transformer for better handling of skewed distributions
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    X_scaled_array = pt.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Label mapping:", mapping)
    
    return X_scaled, y_encoded, pt

def print_dataset_stats(y, output_dir=config.OUTPUT_DIR):
    """Print comprehensive dataset statistics"""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    total = len(y)
    trojans = (y == 1).sum()
    normal = (y == 0).sum()
    
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {total}")
    print(f"Normal samples: {normal} ({normal/total*100:.2f}%)")
    print(f"Trojan samples: {trojans} ({trojans/total*100:.2f}%)")
    print(f"Class imbalance ratio: {normal/trojans:.2f}:1")
    
    # Class balance visualization
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y, palette="Set1")
    plt.title("Class Distribution")
    plt.xlabel("Class (0=Normal, 1=Trojan)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()
    
    print(f"Saved class distribution plot to {output_dir}")

def split_data(X, y):
    """Split data into training, validation, and test sets"""
    # First split out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # Then create a validation set from the remaining data
    valid_size_adjusted = config.VALIDATION_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=valid_size_adjusted, 
        random_state=config.RANDOM_STATE, stratify=y_temp
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def handle_class_imbalance(X_train, y_train):
    """Apply SMOTE to handle class imbalance"""
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Training samples: {X_train_res.shape[0]}")
    print(f"Class distribution after SMOTE: {np.bincount(y_train_res)}")
    
    return X_train_res, y_train_res