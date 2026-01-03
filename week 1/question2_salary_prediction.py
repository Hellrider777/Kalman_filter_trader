"""
Question 2: Salary Prediction & Bias Detection
This script implements comprehensive bias detection and fairness analysis
for salary prediction models including:
- EDA and data preprocessing
- OLS Linear Regression
- Multiple fairness metrics
- SHAP analysis for interpretability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# ============================================================================
# Part 1: Exploratory Data Analysis (EDA)
# ============================================================================

def perform_eda(df, save_prefix='eda'):
    """
    Perform comprehensive exploratory data analysis
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataset
    save_prefix : str
        Prefix for saved plots
    """
    print("\n" + "="*70)
    print("PART 1: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Basic statistics
    print("\n1. Dataset Shape:")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n2. Column Types:")
    print(df.dtypes)
    
    print("\n3. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   No missing values!")
    
    print("\n4. Basic Statistics:")
    print(df.describe())
    
    # Univariate analysis for numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print(f"\n5. Univariate Statistics for Numeric Features:")
    for col in numeric_cols:
        print(f"\n   {col}:")
        print(f"      Mean: {df[col].mean():.2f}")
        print(f"      Median: {df[col].median():.2f}")
        print(f"      Std: {df[col].std():.2f}")
        print(f"      Min: {df[col].min():.2f}")
        print(f"      Max: {df[col].max():.2f}")
        print(f"      Skewness: {df[col].skew():.2f}")
        print(f"      Kurtosis: {df[col].kurtosis():.2f}")
    
    # Categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print(f"\n6. Categorical Features Distribution:")
    for col in categorical_cols:
        print(f"\n   {col}:")
        print(df[col].value_counts())
    
    # Visualizations
    print("\n7. Creating Visualizations...")
    
    # Histograms for numeric features
    n_numeric = len(numeric_cols)
    if n_numeric > 0:
        fig, axes = plt.subplots((n_numeric + 2) // 3, 3, figsize=(15, 5 * ((n_numeric + 2) // 3)))
        axes = axes.flatten() if n_numeric > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_numeric, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_histograms.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved histograms to {save_prefix}_histograms.png")
        plt.close()
    
    # Salary distribution by categorical variables
    if 'salary' in df.columns:
        categorical_for_analysis = ['gender', 'education_level', 'industry'] if all(col in df.columns for col in ['gender', 'education_level', 'industry']) else categorical_cols[:3]
        
        fig, axes = plt.subplots(1, len(categorical_for_analysis), figsize=(15, 5))
        if len(categorical_for_analysis) == 1:
            axes = [axes]
        
        for idx, col in enumerate(categorical_for_analysis):
            if col in df.columns:
                df.boxplot(column='salary', by=col, ax=axes[idx])
                axes[idx].set_title(f'Salary by {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Salary')
                plt.sca(axes[idx])
                plt.xticks(rotation=45)
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig('salary_distribution.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved salary distribution plots to salary_distribution.png")
        plt.close()
    
    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved correlation heatmap to correlation_heatmap.png")
        plt.close()
    
    print("\n   ✓ EDA Complete!")

# ============================================================================
# Part 2 & 3: Data Preprocessing
# ============================================================================

def preprocess_data(df):
    """
    Handle missing values, outliers, and encode categorical features
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataset
        
    Returns:
    --------
    df_processed : pandas DataFrame
        Processed dataset
    encoders : dict
        Dictionary of label encoders for categorical features
    """
    print("\n" + "="*70)
    print("PARTS 2 & 3: DATA PREPROCESSING")
    print("="*70)
    
    df_processed = df.copy()
    
    # Handle missing values
    print("\n1. Handling Missing Values:")
    missing_before = df_processed.isnull().sum().sum()
    print(f"   Total missing values before: {missing_before}")
    
    # Strategy: 
    # - Numeric: Fill with median (robust to outliers)
    # - Categorical: Fill with mode
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"   - Filled {col} with median: {median_val:.2f}")
    
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            mode_val = df_processed[col].mode()[0]
            df_processed[col].fillna(mode_val, inplace=True)
            print(f"   - Filled {col} with mode: {mode_val}")
    
    missing_after = df_processed.isnull().sum().sum()
    print(f"   Total missing values after: {missing_after}")
    
    # Handle outliers
    print("\n2. Handling Outliers (IQR method):")
    for col in numeric_cols:
        if col != 'salary':  # Don't remove outliers from target variable
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
            
            # Cap outliers instead of removing (to preserve sample size)
            df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
            
            if outliers_before > 0:
                print(f"   - {col}: Capped {outliers_before} outliers")
    
    # Encode categorical features
    print("\n3. Encoding Categorical Features:")
    print("   Strategy: Label Encoding for ordinal features, One-Hot for nominal")
    
    encoders = {}
    
    # Ordinal encoding for education_level (has natural order)
    if 'education_level' in df_processed.columns:
        education_order = {'HighSchool': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}
        df_processed['education_level_encoded'] = df_processed['education_level'].map(education_order)
        print(f"   - education_level: Ordinal encoding (HighSchool=0, Bachelors=1, Masters=2, PhD=3)")
    
    # Label encoding for other categorical features
    for col in categorical_cols:
        if col != 'education_level':
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            encoders[col] = le
            print(f"   - {col}: Label encoding ({len(le.classes_)} classes)")
    
    print("\n   Justification:")
    print("   - Label Encoding: Efficient for tree-based models and maintains order")
    print("   - Ordinal for education: Preserves natural ordering")
    print("   - Could use One-Hot for truly nominal features (increases dimensionality)")
    
    return df_processed, encoders

# ============================================================================
# Part 4: Train-Test Split with Stratification
# ============================================================================

def split_data(df, target_col='salary', stratify_col='gender_encoded', test_size=0.2):
    """
    Split data with stratification
    
    Parameters:
    -----------
    df : pandas DataFrame
        Preprocessed dataset
    target_col : str
        Target variable column name
    stratify_col : str
        Column to use for stratification
    test_size : float
        Proportion of test set
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split data
    feature_names : list
        List of feature names
    """
    print("\n" + "="*70)
    print("PART 4: TRAIN-TEST SPLIT WITH STRATIFICATION")
    print("="*70)
    
    # Select feature columns (encoded versions)
    feature_cols = [col for col in df.columns if col.endswith('_encoded') or 
                    (col in df.select_dtypes(include=[np.number]).columns and col != target_col)]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    print(f"\n1. Feature Selection:")
    print(f"   Selected {len(feature_cols)} features")
    print(f"   Features: {feature_cols}")
    
    print(f"\n2. Stratification Strategy:")
    print(f"   Stratifying by: {stratify_col}")
    print(f"   Justification: Ensures balanced representation of gender groups")
    print(f"   This prevents bias in train/test splits and maintains population distribution")
    
    # Perform stratified split
    stratify_values = df[stratify_col].values if stratify_col in df.columns else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_values
    )
    
    print(f"\n3. Split Results:")
    print(f"   Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"   Test set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    
    if stratify_values is not None:
        train_stratify = df.loc[df.index[:len(X_train)], stratify_col]
        print(f"\n4. Stratification Verification:")
        print(f"   Distribution in original data:")
        print(df[stratify_col].value_counts(normalize=True))
    
    return X_train, X_test, y_train, y_test, feature_cols

# ============================================================================
# Parts 5-9: Model Training and Evaluation
# ============================================================================

def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    """
    Train OLS Linear Regression and evaluate
    
    Returns:
    --------
    model : LinearRegression
        Trained model
    y_pred : array
        Predictions on test set
    """
    print("\n" + "="*70)
    print("PARTS 5-9: MODEL TRAINING AND EVALUATION")
    print("="*70)
    
    # Train model
    print("\n1. Training Baseline OLS Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("   ✓ Model trained successfully!")
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Coefficients and statistics
    print("\n2. Model Coefficients:")
    print(f"   Intercept: {model.intercept_:.4f}")
    
    # Compute standard errors
    n = X_train.shape[0]
    p = X_train.shape[1]
    residuals = y_train - y_pred_train
    mse = np.sum(residuals**2) / (n - p - 1)
    
    # Variance-covariance matrix
    X_with_intercept = np.column_stack([np.ones(n), X_train])
    var_covar = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    std_errors = np.sqrt(np.diag(var_covar))
    
    # T-statistics and p-values
    t_stats = np.concatenate([[model.intercept_], model.coef_]) / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
    
    # Confidence intervals (95%)
    ci_lower = np.concatenate([[model.intercept_], model.coef_]) - 1.96 * std_errors
    ci_upper = np.concatenate([[model.intercept_], model.coef_]) + 1.96 * std_errors
    
    print(f"\n   {'Feature':<25} {'Coef':<12} {'Std Err':<12} {'t-stat':<12} {'p-value':<12} {'95% CI':<25}")
    print(f"   {'-'*100}")
    print(f"   {'Intercept':<25} {model.intercept_:<12.4f} {std_errors[0]:<12.4f} {t_stats[0]:<12.4f} {p_values[0]:<12.6f} [{ci_lower[0]:.4f}, {ci_upper[0]:.4f}]")
    
    for i, (feat, coef) in enumerate(zip(feature_names, model.coef_), 1):
        print(f"   {feat:<25} {coef:<12.4f} {std_errors[i]:<12.4f} {t_stats[i]:<12.4f} {p_values[i]:<12.6f} [{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]")
    
    # Interpretation of top 5 influential coefficients
    print("\n3. Top 5 Most Influential Coefficients:")
    coef_importance = list(zip(feature_names, model.coef_))
    coef_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for i, (feat, coef) in enumerate(coef_importance[:5], 1):
        direction = "increases" if coef > 0 else "decreases"
        print(f"   {i}. {feat}: {coef:.4f}")
        print(f"      → A unit increase in {feat} {direction} salary by ${abs(coef):.2f}")
    
    # Performance metrics
    print("\n4. Performance Metrics:")
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"   Training Set:")
    print(f"      RMSE: ${rmse_train:,.2f}")
    print(f"      MAE:  ${mae_train:,.2f}")
    print(f"      R²:   {r2_train:.4f}")
    
    print(f"\n   Test Set:")
    print(f"      RMSE: ${rmse_test:,.2f}")
    print(f"      MAE:  ${mae_test:,.2f}")
    print(f"      R²:   {r2_test:.4f}")
    
    # Check linearity assumptions
    print("\n5. Linearity Assumption Check:")
    
    residuals_test = y_test - y_pred_test
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred_test, residuals_test, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Salary')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted Values')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals_test, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linearity_check.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved linearity check plots to linearity_check.png")
    plt.close()
    
    print("\n   Interpretation:")
    print("   - Residuals should be randomly scattered around zero")
    print("   - Q-Q plot should show points along diagonal for normality")
    print("   - Patterns suggest violations of linear assumptions")
    
    return model, y_pred_test, residuals_test

# ============================================================================
# Part 10: Fairness Metrics
# ============================================================================

def compute_fairness_metrics(df, y_test, y_pred, test_indices, protected_attr='gender'):
    """
    Compute comprehensive fairness metrics
    
    Parameters:
    -----------
    df : pandas DataFrame
        Original dataframe with protected attributes
    y_test : array
        True values
    y_pred : array
        Predicted values
    test_indices : array
        Indices of test samples
    protected_attr : str
        Protected attribute column name
    """
    print("\n" + "="*70)
    print("PART 10: FAIRNESS METRICS AND BIAS DETECTION")
    print("="*70)
    
    # Get protected attribute for test set
    gender = df.loc[df.index[test_indices], protected_attr].values if protected_attr in df.columns else None
    
    if gender is None:
        print(f"\n⚠ Warning: Protected attribute '{protected_attr}' not found in dataset")
        return
    
    # Define groups
    groups = np.unique(gender)
    print(f"\n1. Protected Attribute: {protected_attr}")
    print(f"   Groups: {groups}")
    
    # Compare Male vs Female, and Male vs Other
    comparisons = [
        ('Male', 'Female'),
        ('Male', 'Other')
    ]
    
    results = {}
    
    for group1, group2 in comparisons:
        if group1 not in groups or group2 not in groups:
            print(f"\n⚠ Skipping comparison {group1} vs {group2} (group not found)")
            continue
        
        print(f"\n{'='*70}")
        print(f"COMPARISON: {group1} vs {group2}")
        print(f"{'='*70}")
        
        # Filter data for each group
        mask1 = gender == group1
        mask2 = gender == group2
        
        y_true_g1 = y_test[mask1]
        y_pred_g1 = y_pred[mask1]
        y_true_g2 = y_test[mask2]
        y_pred_g2 = y_pred[mask2]
        
        print(f"\nSample sizes:")
        print(f"   {group1}: {len(y_true_g1)}")
        print(f"   {group2}: {len(y_true_g2)}")
        
        # (a) Mean Salary Prediction Difference
        mean_pred_g1 = np.mean(y_pred_g1)
        mean_pred_g2 = np.mean(y_pred_g2)
        mean_diff = mean_pred_g1 - mean_pred_g2
        
        print(f"\n(a) Mean Salary Prediction Difference:")
        print(f"    Mean predicted salary for {group1}: ${mean_pred_g1:,.2f}")
        print(f"    Mean predicted salary for {group2}: ${mean_pred_g2:,.2f}")
        print(f"    Difference: ${mean_diff:,.2f}")
        print(f"    Interpretation: {group1} predicted ${abs(mean_diff):,.2f} {'higher' if mean_diff > 0 else 'lower'} than {group2}")
        
        # (b) Mean Absolute Error per group
        mae_g1 = mean_absolute_error(y_true_g1, y_pred_g1)
        mae_g2 = mean_absolute_error(y_true_g2, y_pred_g2)
        mae_diff = abs(mae_g1 - mae_g2)
        
        print(f"\n(b) Mean Absolute Error (MAE) per Group:")
        print(f"    MAE for {group1}: ${mae_g1:,.2f}")
        print(f"    MAE for {group2}: ${mae_g2:,.2f}")
        print(f"    Difference: ${mae_diff:,.2f}")
        print(f"    Interpretation: Model is {'more' if mae_g1 > mae_g2 else 'less'} accurate for {group1}")
        
        # (c) Demographic Parity Difference (DPD)
        # Proportion receiving "high salary" (above median)
        threshold = np.median(y_test)
        prop_high_g1 = np.mean(y_pred_g1 > threshold)
        prop_high_g2 = np.mean(y_pred_g2 > threshold)
        dpd = prop_high_g1 - prop_high_g2
        
        print(f"\n(c) Demographic Parity Difference (DPD):")
        print(f"    Threshold (median salary): ${threshold:,.2f}")
        print(f"    Proportion predicted above threshold ({group1}): {prop_high_g1:.4f}")
        print(f"    Proportion predicted above threshold ({group2}): {prop_high_g2:.4f}")
        print(f"    DPD: {dpd:.4f}")
        print(f"    Interpretation: DPD measures if groups receive favorable outcomes at equal rates")
        print(f"                    Value close to 0 indicates demographic parity")
        print(f"                    {'Significant' if abs(dpd) > 0.1 else 'No significant'} disparity detected")
        
        # (d) Equal Opportunity Difference (EOD)
        # True Positive Rate for those who actually have high salary
        high_salary_g1 = y_true_g1 > threshold
        high_salary_g2 = y_true_g2 > threshold
        
        tpr_g1 = np.mean((y_pred_g1 > threshold)[high_salary_g1]) if high_salary_g1.sum() > 0 else 0
        tpr_g2 = np.mean((y_pred_g2 > threshold)[high_salary_g2]) if high_salary_g2.sum() > 0 else 0
        eod = tpr_g1 - tpr_g2
        
        print(f"\n(d) Equal Opportunity Difference (EOD):")
        print(f"    TPR for {group1}: {tpr_g1:.4f}")
        print(f"    TPR for {group2}: {tpr_g2:.4f}")
        print(f"    EOD: {eod:.4f}")
        print(f"    Interpretation: EOD measures if qualified individuals have equal opportunity")
        print(f"                    regardless of protected attribute")
        print(f"                    {'Significant' if abs(eod) > 0.1 else 'No significant'} disparity detected")
        
        # (e) Predictive Equality
        # False Positive Rate for those who don't actually have high salary
        low_salary_g1 = y_true_g1 <= threshold
        low_salary_g2 = y_true_g2 <= threshold
        
        fpr_g1 = np.mean((y_pred_g1 > threshold)[low_salary_g1]) if low_salary_g1.sum() > 0 else 0
        fpr_g2 = np.mean((y_pred_g2 > threshold)[low_salary_g2]) if low_salary_g2.sum() > 0 else 0
        pe_diff = fpr_g1 - fpr_g2
        
        print(f"\n(e) Predictive Equality:")
        print(f"    FPR for {group1}: {fpr_g1:.4f}")
        print(f"    FPR for {group2}: {fpr_g2:.4f}")
        print(f"    Difference: {pe_diff:.4f}")
        print(f"    Interpretation: Predictive equality ensures similar false positive rates")
        print(f"                    {'Significant' if abs(pe_diff) > 0.1 else 'No significant'} disparity detected")
        
        # (f) Disparate Impact Ratio (DIR)
        dir_ratio = prop_high_g2 / prop_high_g1 if prop_high_g1 > 0 else 0
        
        print(f"\n(f) Disparate Impact Ratio (DIR):")
        print(f"    DIR: {dir_ratio:.4f}")
        print(f"    Interpretation: DIR = (favorable outcome rate for {group2}) / (rate for {group1})")
        print(f"                    Ideal value: 1.0 (perfect parity)")
        print(f"                    80% rule: DIR should be ≥ 0.8")
        print(f"                    {'PASS' if dir_ratio >= 0.8 else 'FAIL'} the 80% rule")
        
        # Store results
        results[f"{group1}_vs_{group2}"] = {
            'mean_diff': mean_diff,
            'mae_g1': mae_g1,
            'mae_g2': mae_g2,
            'dpd': dpd,
            'eod': eod,
            'pe': pe_diff,
            'dir': dir_ratio
        }
    
    # Create summary table
    print(f"\n{'='*70}")
    print("FAIRNESS METRICS SUMMARY TABLE")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<35} {'Male vs Female':<20} {'Male vs Other':<20}")
    print(f"{'-'*75}")
    
    for metric_name in ['mean_diff', 'mae_g1', 'mae_g2', 'dpd', 'eod', 'pe', 'dir']:
        row = f"{metric_name:<35}"
        for comp in ['Male_vs_Female', 'Male_vs_Other']:
            if comp in results:
                row += f"{results[comp].get(metric_name, 'N/A'):<20.4f}"
            else:
                row += f"{'N/A':<20}"
        print(row)
    
    return results

# ============================================================================
# Part 10 (continued): Residual Distribution by Gender
# ============================================================================

def plot_residuals_by_gender(df, y_test, y_pred, test_indices, protected_attr='gender'):
    """
    Plot residual distribution by gender
    """
    print("\n" + "="*70)
    print("RESIDUAL DISTRIBUTION BY GENDER")
    print("="*70)
    
    gender = df.loc[df.index[test_indices], protected_attr].values
    residuals = y_test - y_pred
    
    # Create DataFrame for easy plotting
    residual_df = pd.DataFrame({
        'residual': residuals,
        'gender': gender
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    residual_df.boxplot(column='residual', by='gender', ax=axes[0])
    axes[0].set_title('Residual Distribution by Gender')
    axes[0].set_xlabel('Gender')
    axes[0].set_ylabel('Residuals')
    plt.sca(axes[0])
    plt.xticks(rotation=0)
    
    # Histogram
    for gender_val in residual_df['gender'].unique():
        data = residual_df[residual_df['gender'] == gender_val]['residual']
        axes[1].hist(data, alpha=0.6, bins=30, label=gender_val, edgecolor='black')
    
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution by Gender')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('residuals_by_gender.png', dpi=300, bbox_inches='tight')
    print("✓ Saved residual distribution plot to residuals_by_gender.png")
    plt.close()

# ============================================================================
# Part 11: Statistical Testing
# ============================================================================

def statistical_testing(df, y_test, y_pred, test_indices, protected_attr='gender'):
    """
    Conduct statistical tests for bias
    """
    print("\n" + "="*70)
    print("PART 11: STATISTICAL TESTING FOR BIAS")
    print("="*70)
    
    gender = df.loc[df.index[test_indices], protected_attr].values
    residuals = y_test - y_pred
    
    groups = np.unique(gender)
    
    print(f"\nConducting t-tests for mean residual differences:")
    
    # Conduct pairwise t-tests
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            group1 = groups[i]
            group2 = groups[j]
            
            resid1 = residuals[gender == group1]
            resid2 = residuals[gender == group2]
            
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(resid1, resid2)
            
            print(f"\n{group1} vs {group2}:")
            print(f"   Mean residual ({group1}): ${np.mean(resid1):,.2f}")
            print(f"   Mean residual ({group2}): ${np.mean(resid2):,.2f}")
            print(f"   t-statistic: {t_stat:.4f}")
            print(f"   p-value: {p_value:.6f}")
            print(f"   Result: {'Significant' if p_value < 0.05 else 'Not significant'} difference (α=0.05)")
            
            if p_value < 0.05:
                print(f"   ⚠ WARNING: Significant bias detected!")

# ============================================================================
# Part 12: Systematic Bias Analysis
# ============================================================================

def analyze_systematic_bias(df, y_test, y_pred, test_indices, protected_attr='gender'):
    """
    Identify systematic over/underestimation
    """
    print("\n" + "="*70)
    print("PART 12: SYSTEMATIC BIAS ANALYSIS")
    print("="*70)
    
    gender = df.loc[df.index[test_indices], protected_attr].values
    residuals = y_test - y_pred
    
    print(f"\nAnalyzing systematic over/underestimation by {protected_attr}:")
    
    for group in np.unique(gender):
        group_residuals = residuals[gender == group]
        mean_residual = np.mean(group_residuals)
        
        print(f"\n{group}:")
        print(f"   Mean residual: ${mean_residual:,.2f}")
        
        if mean_residual > 1000:
            print(f"   → Model UNDERESTIMATES salary by ${mean_residual:,.2f} on average")
            print(f"   → This suggests systematic bias AGAINST {group}")
        elif mean_residual < -1000:
            print(f"   → Model OVERESTIMATES salary by ${abs(mean_residual):,.2f} on average")
            print(f"   → This suggests systematic bias IN FAVOR OF {group}")
        else:
            print(f"   → No significant systematic bias detected")

# ============================================================================
# Part 13: SHAP Analysis
# ============================================================================

def shap_analysis(model, X_train, X_test, feature_names):
    """
    Perform SHAP analysis for model interpretability
    """
    print("\n" + "="*70)
    print("PART 13: SHAP ANALYSIS")
    print("="*70)
    
    try:
        import shap
        
        print("\n1. Computing SHAP values...")
        
        # Create explainer
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test[:1000])  # Limit for performance
        
        print("   ✓ SHAP values computed!")
        
        # Summary plot
        print("\n2. Creating SHAP summary plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test[:1000], feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved SHAP summary plot to shap_summary.png")
        plt.close()
        
        # Dependence plots for top 3 features
        print("\n3. Creating SHAP dependence plots for top 3 features...")
        
        # Get feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_3_indices = np.argsort(feature_importance)[-3:][::-1]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, feat_idx in enumerate(top_3_indices):
            shap.dependence_plot(feat_idx, shap_values, X_test[:1000], 
                               feature_names=feature_names, ax=axes[idx], show=False)
        
        plt.tight_layout()
        plt.savefig('shap_dependence.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved SHAP dependence plots to shap_dependence.png")
        plt.close()
        
        print("\n4. SHAP Interpretation:")
        print("   - Summary plot shows feature importance and impact direction")
        print("   - Red indicates high feature values, blue indicates low values")
        print("   - Dependence plots show how feature values affect predictions")
        
    except ImportError:
        print("\n⚠ SHAP library not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'shap'])
        print("   Please run the script again to generate SHAP plots")

# ============================================================================
# Main execution
# ============================================================================

def main():
    """
    Main function to execute all parts of Question 2
    """
    print("\n" + "="*70)
    print("QUESTION 2: SALARY PREDICTION & BIAS DETECTION")
    print("="*70)
    
    # Load dataset
    try:
        df = pd.read_csv('salary_dataset.csv')
        print(f"\n✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("\n⚠ Dataset file 'salary_dataset.csv' not found.")
        print("  Generating synthetic data for demonstration...")
        
        # Generate synthetic salary dataset
        np.random.seed(42)
        n_samples = 12000
        
        df = pd.DataFrame({
            'age': np.random.randint(22, 65, n_samples),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.6, 0.35, 0.05]),
            'education_level': np.random.choice(['HighSchool', 'Bachelors', 'Masters', 'PhD'], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
            'years_experience': np.random.randint(0, 40, n_samples),
            'job_title': np.random.choice([f'Job_{i}' for i in range(15)], n_samples),
            'performance_score': np.random.uniform(1, 5, n_samples),
            'industry': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Retail'], n_samples),
            'city': np.random.choice([f'City_{i}' for i in range(12)], n_samples),
            'previous_companies': np.random.randint(0, 8, n_samples),
            'remote_worker': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
        })
        
        # Generate salary with some bias
        education_effect = df['education_level'].map({'HighSchool': 0, 'Bachelors': 15000, 'Masters': 30000, 'PhD': 45000})
        experience_effect = df['years_experience'] * 1000
        performance_effect = df['performance_score'] * 5000
        gender_bias = df['gender'].map({'Male': 5000, 'Female': 0, 'Other': -2000})  # Intentional bias
        
        df['salary'] = (50000 + education_effect + experience_effect + performance_effect + 
                       gender_bias + np.random.normal(0, 5000, n_samples))
        df['salary'] = df['salary'].clip(25000, 200000)
        
        print(f"✓ Generated synthetic dataset: {n_samples} samples")
    
    # Perform EDA
    perform_eda(df)
    
    # Preprocess data
    df_processed, encoders = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test, feature_names = split_data(df_processed)
    
    # Train and evaluate model
    model, y_pred, residuals = train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names)
    
    # Get test indices
    test_size = len(X_test)
    test_indices = list(range(len(df) - test_size, len(df)))
    
    # Fairness metrics
    fairness_results = compute_fairness_metrics(df, y_test, y_pred, test_indices)
    
    # Plot residuals by gender
    plot_residuals_by_gender(df, y_test, y_pred, test_indices)
    
    # Statistical testing
    statistical_testing(df, y_test, y_pred, test_indices)
    
    # Systematic bias analysis
    analyze_systematic_bias(df, y_test, y_pred, test_indices)
    
    # SHAP analysis
    shap_analysis(model, X_train, X_test, feature_names)
    
    print("\n" + "="*70)
    print("QUESTION 2 COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - eda_histograms.png")
    print("  - eda_salary_by_category.png")
    print("  - eda_correlation_heatmap.png")
    print("  - linearity_check.png")
    print("  - residuals_by_gender.png")
    print("  - shap_summary.png")
    print("  - shap_dependence.png")

if __name__ == "__main__":
    main()
