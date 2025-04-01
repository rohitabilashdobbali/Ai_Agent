"""
Model evaluation and performance metrics for Hardware Trojan Detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, precision_recall_curve, auc, recall_score, precision_score,
    roc_curve
)
import os

import config

def find_optimal_threshold(model, X_val, y_val, output_dir=config.OUTPUT_DIR):
    """Find optimal classification threshold using precision-recall curve"""
    y_scores = model.predict_proba(X_val)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_scores)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find threshold that maximizes F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Plot the precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.', label=f'Precision-Recall curve (AUC = {auc(recall, precision):.3f})')
    plt.scatter(recall[optimal_idx], precision[optimal_idx], marker='o', color='red', 
                label=f'Optimal threshold: {optimal_threshold:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve with Optimal Threshold')
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()
    
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"At this threshold - Precision: {precision[optimal_idx]:.3f}, Recall: {recall[optimal_idx]:.3f}")
    
    return optimal_threshold

def evaluate_model(model, X_test, y_test, threshold=0.5, model_name="Model", output_dir=config.OUTPUT_DIR):
    """Comprehensive model evaluation with visualization"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\n=== {model_name} Evaluation ===")
    
    # Get predictions
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    # Print classification report
    print(f"Classification threshold: {threshold:.3f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall (Sensitivity): {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_validation_results.png"))
    plt.close()
    
    return cv_results

def generate_report(comparison_df, top_features, threshold, execution_time, output_dir=config.OUTPUT_DIR):
    """Generate a comprehensive report on the Trojan detection results"""
    from datetime import datetime
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create report string
    report = []
    report.append("======================================================")
    report.append("      HARDWARE TROJAN DETECTION SYSTEM REPORT         ")
    report.append("======================================================")
    report.append(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total execution time: {execution_time:.2f} minutes")
    report.append("\n1. MODEL PERFORMANCE SUMMARY")
    report.append("------------------------------------------------------")
    report.append(comparison_df.to_string(index=False))
    
    # Add best model information
    best_model_idx = comparison_df['F1 Score'].idxmax()
    best_model = comparison_df.iloc[best_model_idx]
    report.append("\n2. BEST PERFORMING MODEL")
    report.append("------------------------------------------------------")
    report.append(f"Model: {best_model['Model']}")
    report.append(f"F1 Score: {best_model['F1 Score']:.4f}")
    report.append(f"Balanced Accuracy: {best_model['Balanced Accuracy']:.4f}")
    report.append(f"Precision: {best_model['Precision']:.4f}")
    report.append(f"Recall: {best_model['Recall']:.4f}")
    report.append(f"Classification Threshold: {threshold:.3f}")
    
    # Add feature importance information
    report.append("\n3. TOP PREDICTIVE FEATURES")
    report.append("------------------------------------------------------")
    for i, feature in enumerate(top_features):
        report.append(f"{i+1}. {feature}")
    
    # Add recommendations
    report.append("\n4. RECOMMENDATIONS")
    report.append("------------------------------------------------------")
    report.append("Based on the analysis, we recommend the following actions:")
    report.append("1. Focus hardware monitoring on the top predictive features")
    report.append("2. Implement real-time detection using the ensemble model")
    report.append("3. Set alert thresholds based on the optimal classification threshold")
    report.append("4. Schedule periodic model retraining with new data")
    
    # Add footnotes
    report.append("\n5. NOTES")
    report.append("------------------------------------------------------")
    report.append("- This model is designed for detection of hardware Trojans in circuits")
    report.append("- Performance may vary with different hardware configurations")
    report.append("- Regular updates to the model are recommended as new Trojan patterns emerge")
    
    # Save report to file
    report_str = "\n".join(report)
    report_path = os.path.join(output_dir, "trojan_detection_report.txt")
    with open(report_path, "w") as f:
        f.write(report_str)
    
    print(f"Report generated and saved to '{report_path}'")
    return report_stros.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()
    
    # ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_roc_curve.png"))
    plt.close()
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def compare_models(metrics_dict, output_dir=config.OUTPUT_DIR):
    """Compare performance of multiple models"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    comparison_data = []
    
    for model_name, metrics in metrics_dict.items():
        comparison_data.append({
            "Model": model_name,
            "Accuracy": metrics['accuracy'],
            "Balanced Accuracy": metrics['balanced_accuracy'],
            "Precision": metrics['precision'],
            "Recall": metrics['recall'],
            "F1 Score": metrics['f1'],
            "ROC AUC": metrics['roc_auc']
        })
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Visualize model comparison
    plt.figure(figsize=(14, 8))
    metrics_to_plot = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i+1)
        sns.barplot(x='Model', y=metric, data=comparison_df)
        plt.title(metric)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()
    
    return comparison_df

def extract_top_features(models, feature_names, output_dir=config.OUTPUT_DIR):
    """Extract and combine top features from tree-based models"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get combined feature importance from tree-based models
    feature_importance = np.zeros(len(feature_names))
    n_models = 0
    
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            feature_importance += model.feature_importances_
            n_models += 1
    
    if n_models == 0:
        print("No models with feature importance found")
        return []
    
    # Average the feature importances
    feature_importance /= n_models
    
    # Sort features by importance
    indices = np.argsort(feature_importance)[::-1]
    
    # Plot combined feature importance
    plt.figure(figsize=(12, 8))
    plt.title("Combined Feature Importance")
    plt.barh(range(min(20, len(indices))), feature_importance[indices[:20]], align='center')
    plt.yticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]])
    plt.xlabel('Average Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_feature_importance.png"))
    plt.close()
    
    # Get top features
    top_features = [feature_names[i] for i in indices[:10]]
    
    # Print top features
    print("\nTop 10 most important features:")
    for i, feature in enumerate(top_features):
        print(f"{i+1}. {feature}: {feature_importance[indices[i]]:.4f}")
    
    return top_features

def run_cross_validation(data, feature_names, n_folds=5, output_dir=config.OUTPUT_DIR):
    """Run cross-validation to validate model robustness"""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    print("\n=== Cross-Validation for Model Robustness ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data - assuming data has a 'Label' column
    X = data.drop('Label', axis=1) if 'Label' in data.columns else data
    y = data['Label'] if 'Label' in data.columns else None
    
    if y is None:
        print("No 'Label' column found in data. Cross-validation requires labeled data.")
        return None
    
    # Define models to test
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=config.RANDOM_STATE,
            use_label_encoder=False
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=config.RANDOM_STATE,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=config.RANDOM_STATE
        )
    }
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
    
    # Track results
    cv_results = {}
    
    # Run cross-validation for each model
    for name, model in models.items():
        print(f"\nRunning {n_folds}-fold cross-validation for {name}...")
        
        # Calculate multiple metrics
        accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        balanced_acc = cross_val_score(model, X, y, cv=cv, scoring='balanced_accuracy')
        precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
        recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
        f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
        roc_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        
        # Store results
        cv_results[name] = {
            'accuracy': {
                'mean': accuracy.mean(),
                'std': accuracy.std(),
                'values': accuracy.tolist()
            },
            'balanced_accuracy': {
                'mean': balanced_acc.mean(),
                'std': balanced_acc.std(),
                'values': balanced_acc.tolist()
            },
            'precision': {
                'mean': precision.mean(),
                'std': precision.std(),
                'values': precision.tolist()
            },
            'recall': {
                'mean': recall.mean(),
                'std': recall.std(),
                'values': recall.tolist()
            },
            'f1': {
                'mean': f1.mean(),
                'std': f1.std(),
                'values': f1.tolist()
            },
            'roc_auc': {
                'mean': roc_auc.mean(),
                'std': roc_auc.std(),
                'values': roc_auc.tolist()
            }
        }
        
        # Print results
        print(f"  Accuracy: {accuracy.mean():.4f} ± {accuracy.std():.4f}")
        print(f"  Balanced Accuracy: {balanced_acc.mean():.4f} ± {balanced_acc.std():.4f}")
        print(f"  Precision: {precision.mean():.4f} ± {precision.std():.4f}")
        print(f"  Recall: {recall.mean():.4f} ± {recall.std():.4f}")
        print(f"  F1 Score: {f1.mean():.4f} ± {f1.std():.4f}")
        print(f"  ROC AUC: {roc_auc.mean():.4f} ± {roc_auc.std():.4f}")
    
    # Plot results comparison
    plt.figure(figsize=(12, 8))
    
    metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_names = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 3, i+1)
        
        means = [cv_results[model][metric]['mean'] for model in models.keys()]
        stds = [cv_results[model][metric]['std'] for model in models.keys()]
        
        bars = plt.bar(
            range(len(models)), 
            means, 
            yerr=stds, 
            capsize=10, 
            alpha=0.7,
            color=['lightblue', 'lightgreen', 'lightcoral']
        )
        
        plt.xticks(range(len(models)), models.keys(), rotation=45)
        plt.title(metric_name)
        plt.ylim(0.5, 1.0)  # Adjust as needed
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, means):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha='center',
                fontsize=8
            )
    
    plt.tight_layout()
    plt.savefig("""Model evaluation and performance metrics for Hardware Trojan Detection""")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, precision_recall_curve, auc, recall_score, precision_score,roc_curve)
import os

import config

def find_optimal_threshold(model, X_val, y_val):
    """Find optimal classification threshold using precision-recall curve"""
    y_scores = model.predict_proba(X_val)[:, 1]
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_scores)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)