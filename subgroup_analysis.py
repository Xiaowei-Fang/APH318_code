import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.utils import resample
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def calculate_confidence_interval(y_true, y_pred, metric_func, n_bootstraps=100):
    """
    Calculate 95% Confidence Interval using Bootstrapping.
    Standard requirement for SCI papers.
    """
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        # Bootstrap sample
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true.iloc[indices])) < 2:
            continue
        score = metric_func(y_true.iloc[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # 95% CI
    lower = sorted_scores[int(0.025 * len(sorted_scores))]
    upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return lower, upper

def run_subgroup_analysis():
    print(">>> Step 1: Data Preparation...")
    df = pd.read_csv('processed_diabetic_data.csv')
    
    # Keep a copy of original features for subgroup slicing before transformation
    X_original = df.drop('diabetes2', axis=1)
    y = df['diabetes2']

    # Generate Polynomial Features (same as the final model)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_original)
    feature_names = poly.get_feature_names_out(X_original.columns)
    X_poly = pd.DataFrame(X_poly, columns=feature_names)

    # Split Data (Stratified)
    # We need indices to track which row belongs to which subgroup
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_poly, y, X_original.index, test_size=0.2, random_state=42, stratify=y
    )
    
    # Recover original features for the test set using indices
    X_test_original = X_original.loc[idx_test]

    # Calculate Class Weight
    weight_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)

    print(">>> Step 2: Training Best Model (XGBoost)...")
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=weight_ratio,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predict probabilities on the full test set
    y_prob_full = model.predict_proba(X_test)[:, 1]

    print(">>> Step 3: Performing Subgroup Analysis...")
    
    # Define Subgroups
    # Note: 'age' is 0-9. >=6 means 60-100 years old.
    subgroups = {
        'Overall': (pd.Series([True]*len(X_test), index=X_test.index), "All Patients"),
        'Gender: Female': (X_test_original['gender'] == 0, "Female"),
        'Gender: Male': (X_test_original['gender'] == 1, "Male"),
        'Age: < 60': (X_test_original['age'] < 6, "Age < 60"),
        'Age: >= 60': (X_test_original['age'] >= 6, "Age >= 60"),
        'Heart Disease: No': (X_test_original['heart_disease'] == 0, "No Heart Disease"),
        'Heart Disease: Yes': (X_test_original['heart_disease'] == 1, "Has Heart Disease"),
        'Blood Pressure: No': (X_test_original['blood_pressure'] == 0, "Normal BP"),
        'Blood Pressure: Yes': (X_test_original['blood_pressure'] == 1, "High BP")
    }

    results = []

    for key, (mask, label) in subgroups.items():
        # Filter data for this subgroup
        # Align mask with y_test indices
        current_mask = mask.values
        if np.sum(current_mask) < 50: # Skip too small groups
            continue
            
        y_sub_true = y_test[current_mask]
        y_sub_prob = y_prob_full[current_mask]
        
        # Calculate AUC
        auc = roc_auc_score(y_sub_true, y_sub_prob)
        
        # Calculate 95% CI
        lower, upper = calculate_confidence_interval(y_sub_true, y_sub_prob, roc_auc_score)
        
        results.append({
            'Subgroup': label,
            'N': len(y_sub_true),
            'AUC': auc,
            'Lower': lower,
            'Upper': upper
        })
        print(f"Processed {label}: N={len(y_sub_true)}, AUC={auc:.4f} ({lower:.4f}-{upper:.4f})")

    df_results = pd.DataFrame(results)

    print(">>> Step 4: Generating Forest Plot...")
    
    # Plotting
    plt.figure(figsize=(10, 7))
    
    # Reverse order for plotting top-down
    df_plot = df_results.iloc[::-1]
    
    y_pos = np.arange(len(df_plot))
    
    # Draw Error Bars
    plt.errorbar(
        x=df_plot['AUC'], 
        y=y_pos, 
        xerr=[df_plot['AUC'] - df_plot['Lower'], df_plot['Upper'] - df_plot['AUC']],
        fmt='o', 
        color='black', 
        ecolor='gray', 
        elinewidth=2, 
        capsize=4,
        label='AUC (95% CI)'
    )
    
    # Add vertical line for Overall AUC
    overall_auc = df_results.iloc[0]['AUC']
    plt.axvline(x=overall_auc, color='red', linestyle='--', linewidth=1, label=f'Overall AUC ({overall_auc:.3f})')
    
    # Formatting
    plt.yticks(y_pos, [f"{row['Subgroup']} (N={row['N']})" for _, row in df_plot.iterrows()], fontsize=11)
    plt.xlabel('Area Under Curve (AUC)', fontsize=12)
    plt.title('Subgroup Analysis: Model Discrimination Performance', fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    # Add value annotations
    for i, (_, row) in enumerate(df_plot.iterrows()):
        plt.text(row['Upper'] + 0.005, i, f"{row['AUC']:.3f}", va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('Final_Forest_Plot.png', dpi=300)
    print("Saved: Final_Forest_Plot.png")
    
    # Print results for paper table
    print("\n--- Subgroup Analysis Table ---")
    print(df_results[['Subgroup', 'N', 'AUC', 'Lower', 'Upper']])

if __name__ == "__main__":
    run_subgroup_analysis()