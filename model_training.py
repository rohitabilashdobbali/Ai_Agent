"""
Model training and hyperparameter tuning for Hardware Trojan Detection
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint, loguniform
import os
import joblib

import config

def tune_xgb_hyperparameters(X_train, y_train):
    """Tune XGBoost hyperparameters with comprehensive search"""
    start_time = time.time()
    print("\n=== Tuning XGBoost Hyperparameters ===")
    
    # Calculate class weights
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
    print(f"Class balance ratio: {scale_pos_weight:.2f}")
    
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        random_state=config.RANDOM_STATE,
        eval_metric=["logloss", "auc", "error"],
        use_label_encoder=False
    )
    
    # Comprehensive parameter grid
    params = {
        "n_estimators": randint(100, 1000),
        "max_depth": randint(3, 12),
        "learning_rate": loguniform(0.001, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "colsample_bylevel": uniform(0.6, 0.4),
        "gamma": uniform(0, 5),
        "min_child_weight": randint(1, 10),
        "scale_pos_weight": [scale_pos_weight],
        "reg_alpha": loguniform(1e-5, 1),
        "reg_lambda": loguniform(1e-5, 1)
    }
    
    cv = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rs = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=params,
        n_iter=config.N_ITER_SEARCH,
        scoring="balanced_accuracy",
        cv=cv,
        verbose=1,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True
    )
    
    rs.fit(X_train, y_train)
    
    # Print results
    duration = time.time() - start_time
    print(f"XGBoost tuning completed in {duration/60:.2f} minutes")
    print(f"Best balanced accuracy: {rs.best_score_:.4f}")
    print("Best parameters:", rs.best_params_)
    
    # Check for overfitting
    train_score = rs.cv_results_['mean_train_score'][rs.best_index_]
    test_score = rs.best_score_
    print(f"Training score: {train_score:.4f}")
    print(f"Cross-validation score: {test_score:.4f}")
    print(f"Difference (overfitting gap): {train_score - test_score:.4f}")
    
    return rs.best_estimator_

def tune_rf_hyperparameters(X_train, y_train):
    """Tune Random Forest hyperparameters"""
    start_time = time.time()
    print("\n=== Tuning Random Forest Hyperparameters ===")
    
    # Class weight can help with imbalanced datasets
    rf = RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced')
    
    params = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    }
    
    cv = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rs = RandomizedSearchCV(
        estimator=rf,
        param_distributions=params,
        n_iter=config.N_ITER_SEARCH,
        scoring="balanced_accuracy",
        cv=cv,
        verbose=1,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True
    )
    
    rs.fit(X_train, y_train)
    
    # Print results
    duration = time.time() - start_time
    print(f"Random Forest tuning completed in {duration/60:.2f} minutes")
    print(f"Best balanced accuracy: {rs.best_score_:.4f}")
    print("Best parameters:", rs.best_params_)
    
    # Check for overfitting
    train_score = rs.cv_results_['mean_train_score'][rs.best_index_]
    test_score = rs.best_score_
    print(f"Training score: {train_score:.4f}")
    print(f"Cross-validation score: {test_score:.4f}")
    print(f"Difference (overfitting gap): {train_score - test_score:.4f}")
    
    return rs.best_estimator_

def tune_gb_hyperparameters(X_train, y_train):
    """Tune Gradient Boosting hyperparameters"""
    start_time = time.time()
    print("\n=== Tuning Gradient Boosting Hyperparameters ===")
    
    gb = GradientBoostingClassifier(random_state=config.RANDOM_STATE)
    
    params = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 10),
        "learning_rate": loguniform(0.001, 0.3),
        "subsample": uniform(0.7, 0.3),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None]
    }
    
    cv = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rs = RandomizedSearchCV(
        estimator=gb,
        param_distributions=params,
        n_iter=config.N_ITER_SEARCH,
        scoring="balanced_accuracy",
        cv=cv,
        verbose=1,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True
    )
    
    rs.fit(X_train, y_train)
    
    # Print results
    duration = time.time() - start_time
    print(f"Gradient Boosting tuning completed in {duration/60:.2f} minutes")
    print(f"Best balanced accuracy: {rs.best_score_:.4f}")
    print("Best parameters:", rs.best_params_)
    
    return rs.best_estimator_

def build_svm_model(X_train, y_train):
    """Build an SVM model with grid search"""
    print("\n=== Building SVM Model ===")
    
    # For larger datasets, we use a smaller parameter grid to keep computation reasonable
    params = {
        "C": loguniform(0.1, 10),
        "gamma": loguniform(0.001, 1),
        "kernel": ["rbf"],
        "class_weight": ["balanced"]
    }
    
    svm = SVC(probability=True, random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    
    rs = RandomizedSearchCV(
        estimator=svm,
        param_distributions=params,
        n_iter=20,  # Fewer iterations for SVM due to computational intensity
        scoring="balanced_accuracy",
        cv=cv,
        verbose=1,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    rs.fit(X_train, y_train)
    print(f"Best SVM score: {rs.best_score_:.4f}")
    print("Best SVM parameters:", rs.best_params_)
    
    return rs.best_estimator_

def plot_feature_importance(model, feature_names, title, filename):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot the feature importances
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.barh(range(min(20, len(indices))), importances[indices[:20]], align='center')
        plt.yticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        # Print top features
        print(f"\nTop 10 features for {title}:")
        for i in range(min(10, len(indices))):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    else:
        print(f"Model {title} doesn't support feature importance visualization")

def build_ensemble_model(models):
    """Build an ensemble model from the best individual models"""
    # Create estimators list
    estimators = []
    for name, model in models.items():
        estimators.append((name, model))
    
    # Create voting classifier
    voting_ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )
    
    return voting_ensemble

def train_models(X_train_res, y_train_res):
    """Train all models and return the results"""
    models = {}
    
    # XGBoost
    print("Training XGBoost model...")
    models['xgb'] = tune_xgb_hyperparameters(X_train_res, y_train_res)
    
    # Random Forest
    print("Training Random Forest model...")
    models['rf'] = tune_rf_hyperparameters(X_train_res, y_train_res)
    
    # Gradient Boosting
    print("Training Gradient Boosting model...")
    models['gb'] = tune_gb_hyperparameters(X_train_res, y_train_res)
    
    # SVM (if dataset is not too large)
    if X_train_res.shape[0] < 10000:  # Only for smaller datasets
        print("Training SVM model...")
        models['svm'] = build_svm_model(X_train_res, y_train_res)
        svm_included = True
    else:
        print("Dataset too large for SVM, excluding from ensemble")
        svm_included = False
    
    return models, svm_included

def save_models(models, ensemble_model, scaler, output_dir=config.OUTPUT_DIR):
    """Save all trained models and the scaler"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save individual models
    for name, model in models.items():
        model_path = os.path.join(output_dir, f"{name}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Saved {name} model to {model_path}")
    
    # Save ensemble model
    ensemble_path = os.path.join(output_dir, "ensemble_model.pkl")
    joblib.dump(ensemble_model, ensemble_path)
    print(f"Saved ensemble model to {ensemble_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    return {
        "ensemble_path": ensemble_path,
        "scaler_path": scaler_path
    }