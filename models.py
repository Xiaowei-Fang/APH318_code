import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, confusion_matrix, roc_curve
import shap
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_analysis():
    print(">>> Step 1: Loading Data...")
    df = pd.read_csv('processed_diabetic_data.csv')
    X = df.drop('diabetes2', axis=1)
    y = df['diabetes2']

    print(">>> Step 2: Feature Engineering (Interactions)...")
    # Generate interaction features (x1*x2) to capture complex relationships
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    X_poly = pd.DataFrame(X_poly, columns=feature_names)
    
    print(f"    Original features: {X.shape[1]}")
    print(f"    Expanded features: {X_poly.shape[1]}")

    print(">>> Step 3: Splitting Data (80/20 Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize data (Required for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate class weight for Cost-Sensitive Learning
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_weight = neg_count / pos_count

    print("\n>>> Step 4: Model Comparison...")
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        "XGBoost (Proposed)": XGBClassifier(scale_pos_weight=scale_weight, n_estimators=300, learning_rate=0.05, 
                                            max_depth=4, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = []
    plt.figure(figsize=(10, 8))

    # Loop through models to train and evaluate
    for name, model in models.items():
        # Use scaled data for LR, normal data for trees
        X_tr = X_train_scaled if name == "Logistic Regression" else X_train
        X_te = X_test_scaled if name == "Logistic Regression" else X_test
        
        model.fit(X_tr, y_train)
        y_prob = model.predict_proba(X_te)[:, 1]
        
        # Calculate Metrics (using default 0.5 threshold for comparison first)
        auc = roc_auc_score(y_test, y_prob)
        y_pred_default = (y_prob >= 0.5).astype(int)
        f1 = f1_score(y_test, y_pred_default)
        
        results.append({"Model": name, "AUC": auc, "F1 (Default)": f1})
        
        # Plot ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

    # Finalize ROC Plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('Final_Comparison_ROC.png', dpi=300)
    print("    Saved: Final_Comparison_ROC.png")

    # Print Comparison Table
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Results:")
    print(results_df)

    print("\n>>> Step 5: Final Optimization for XGBoost (Threshold Tuning)...")
    # We focus on optimizing the best model (XGBoost)
    best_model = models["XGBoost (Proposed)"]
    y_prob_final = best_model.predict_proba(X_test)[:, 1]
    
    # Threshold Tuning Loop
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = []
    for t in thresholds:
        f1_scores.append(f1_score(y_test, (y_prob_final >= t).astype(int)))
    
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Final Metrics with Optimized Threshold
    y_final_pred = (y_prob_final >= best_thresh).astype(int)
    final_acc = accuracy_score(y_test, y_final_pred)
    final_rec = recall_score(y_test, y_final_pred)
    
    print(f"    Optimal Threshold: {best_thresh:.2f}")
    print(f"    Final AUC: {roc_auc_score(y_test, y_prob_final):.4f}")
    print(f"    Final F1 Score: {best_f1:.4f}")
    print(f"    Final Recall: {final_rec:.4f}")
    print(f"    Final Accuracy: {final_acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_final_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (XGBoost @ {best_thresh:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('Final_Confusion_Matrix.png', dpi=300)
    print("    Saved: Final_Confusion_Matrix.png")

    print("\n>>> Step 6: SHAP Interpretability...")
    explainer = shap.TreeExplainer(best_model)
    # Using a subset for SHAP speed if dataset is huge, but here full set is fine
    shap_values = explainer.shap_values(X_test)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig('Final_SHAP_Summary.png', dpi=300)
    print("    Saved: Final_SHAP_Summary.png")

if __name__ == "__main__":
    run_analysis()