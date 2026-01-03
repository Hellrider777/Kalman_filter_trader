"""
Question 1: Linear Regression
This script implements all parts of Question 1 including:
- Computing OLS estimator from scratch
- Comparing with sklearn
- Residual analysis
- Q-Q plots
- Leverage and Cook's distance analysis
- Multicollinearity demonstration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Part 7: Computing OLS Estimator
# ============================================================================

def compute_ols_manually(X, y):
    """
    Compute OLS estimator using the formula: β_hat = (X^T X)^{-1} X^T y
    
    Parameters:
    -----------
    X : numpy array of shape (n, p+1)
        Design matrix with intercept column
    y : numpy array of shape (n,)
        Response variable
        
    Returns:
    --------
    beta_hat : numpy array of shape (p+1,)
        OLS coefficient estimates
    """
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty = X.T @ y
    beta_hat = XtX_inv @ Xty
    return beta_hat

def compare_with_sklearn(X, y):
    """
    Compare manual OLS computation with sklearn's LinearRegression
    
    Parameters:
    -----------
    X : numpy array of shape (n, p)
        Design matrix without intercept
    y : numpy array of shape (n,)
        Response variable
        
    Returns:
    --------
    manual_coef : numpy array
        Manually computed coefficients
    sklearn_coef : numpy array
        Sklearn computed coefficients
    """
    # Add intercept column for manual computation
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    
    # Manual computation
    manual_beta = compute_ols_manually(X_with_intercept, y)
    
    # Sklearn computation
    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)
    sklearn_beta = np.concatenate([[sklearn_model.intercept_], sklearn_model.coef_])
    
    print("=" * 70)
    print("PART 7: OLS ESTIMATOR COMPARISON")
    print("=" * 70)
    print("\nManual OLS coefficients:")
    print(f"Intercept: {manual_beta[0]:.6f}")
    for i, coef in enumerate(manual_beta[1:], 1):
        print(f"β_{i}: {coef:.6f}")
    
    print("\nSklearn LinearRegression coefficients:")
    print(f"Intercept: {sklearn_beta[0]:.6f}")
    for i, coef in enumerate(sklearn_beta[1:], 1):
        print(f"β_{i}: {coef:.6f}")
    
    print("\nDifference (should be near zero):")
    print(f"Max absolute difference: {np.max(np.abs(manual_beta - sklearn_beta)):.10e}")
    
    return manual_beta, sklearn_beta

# ============================================================================
# Part 8: Residuals vs Fitted Values (Homoscedasticity Check)
# ============================================================================

def plot_residuals_vs_fitted(y_true, y_pred, save_path='residuals_vs_fitted.png'):
    """
    Plot residuals vs fitted values to check for homoscedasticity
    
    Parameters:
    -----------
    y_true : numpy array
        True response values
    y_pred : numpy array
        Predicted response values
    save_path : str
        Path to save the plot
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Fitted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residuals vs Fitted Values', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add LOWESS smoothing line to detect patterns
    from scipy.interpolate import UnivariateSpline
    sorted_indices = np.argsort(y_pred)
    try:
        spl = UnivariateSpline(y_pred[sorted_indices], residuals[sorted_indices], s=len(y_pred))
        y_smooth = spl(y_pred[sorted_indices])
        plt.plot(y_pred[sorted_indices], y_smooth, 'b-', linewidth=2, label='Smoothed trend')
        plt.legend()
    except:
        pass
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved residuals vs fitted plot to: {save_path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("PART 8: HOMOSCEDASTICITY CHECK")
    print("=" * 70)
    print("\nInterpretation:")
    print("- If residuals are randomly scattered around zero with constant spread,")
    print("  homoscedasticity holds (constant variance assumption is satisfied).")
    print("- Patterns like funneling or curvature suggest violations.")

# ============================================================================
# Part 9: Q-Q Plot (Normality Check)
# ============================================================================

def plot_qq_plot(residuals, save_path='qq_plot.png'):
    """
    Create Q-Q plot to check normality of residuals
    
    Parameters:
    -----------
    residuals : numpy array
        Residuals from the model
    save_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals', fontsize=14, fontweight='bold')
    plt.xlabel('Theoretical Quantiles', fontsize=12)
    plt.ylabel('Sample Quantiles', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Q-Q plot to: {save_path}")
    plt.close()
    
    # Perform Shapiro-Wilk test for normality
    if len(residuals) <= 5000:
        stat, p_value = stats.shapiro(residuals)
        print("\n" + "=" * 70)
        print("PART 9: NORMALITY CHECK")
        print("=" * 70)
        print(f"\nShapiro-Wilk Test:")
        print(f"  Test statistic: {stat:.6f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Result: {'Reject' if p_value < 0.05 else 'Fail to reject'} normality (α=0.05)")
    
    print("\nInterpretation:")
    print("- If points lie close to the diagonal line, residuals are approximately normal.")
    print("- Deviations from the line suggest non-normality.")

# ============================================================================
# Part 10: Model Assumption Violations
# ============================================================================

def analyze_assumptions(residuals, y_pred):
    """
    Comprehensive analysis of model assumption violations
    """
    print("\n" + "=" * 70)
    print("PART 10: MODEL ASSUMPTION VIOLATIONS")
    print("=" * 70)
    
    print("\n1. LINEARITY:")
    print("   - Check residuals vs fitted plot for patterns")
    print("   - Random scatter suggests linearity holds")
    
    print("\n2. INDEPENDENCE:")
    print("   - Durbin-Watson test (for time series data)")
    from statsmodels.stats.stattools import durbin_watson
    dw_stat = durbin_watson(residuals)
    print(f"   - Durbin-Watson statistic: {dw_stat:.4f}")
    print(f"   - Values near 2 suggest no autocorrelation")
    
    print("\n3. HOMOSCEDASTICITY:")
    print("   - Check residuals vs fitted plot for constant spread")
    print("   - Breusch-Pagan test:")
    # Simple variance ratio test
    n = len(residuals)
    first_half_var = np.var(residuals[:n//2])
    second_half_var = np.var(residuals[n//2:])
    print(f"   - First half variance: {first_half_var:.4f}")
    print(f"   - Second half variance: {second_half_var:.4f}")
    print(f"   - Ratio: {second_half_var/first_half_var:.4f}")
    
    print("\n4. NORMALITY:")
    print("   - Check Q-Q plot")
    print("   - Shapiro-Wilk test (see above)")
    
    print("\n5. MULTICOLLINEARITY:")
    print("   - Check VIF (Variance Inflation Factor)")
    print("   - See Problem 14 for detailed analysis")

# ============================================================================
# Part 11: Leverage and Cook's Distance
# ============================================================================

def compute_leverage_and_cooks_distance(X, y, save_path='leverage_cooks.png'):
    """
    Compute hat matrix, leverage values, and Cook's distance
    
    Parameters:
    -----------
    X : numpy array of shape (n, p+1)
        Design matrix with intercept
    y : numpy array
        Response variable
    save_path : str
        Path to save the plot
    """
    n, p = X.shape
    
    # Compute hat matrix H = X(X^T X)^{-1}X^T
    XtX_inv = np.linalg.inv(X.T @ X)
    H = X @ XtX_inv @ X.T
    
    # Leverage values (diagonal of hat matrix)
    leverage = np.diag(H)
    
    # Compute predictions and residuals
    beta_hat = compute_ols_manually(X, y)
    y_pred = X @ beta_hat
    residuals = y - y_pred
    
    # Compute MSE
    mse = np.sum(residuals**2) / (n - p)
    
    # Standardized residuals
    std_residuals = residuals / np.sqrt(mse * (1 - leverage))
    
    # Cook's distance
    cooks_d = (std_residuals**2 / p) * (leverage / (1 - leverage))
    
    print("\n" + "=" * 70)
    print("PART 11: LEVERAGE AND INFLUENTIAL POINTS")
    print("=" * 70)
    
    # Identify high leverage points (rule of thumb: h_i > 2p/n)
    leverage_threshold = 2 * p / n
    high_leverage = np.where(leverage > leverage_threshold)[0]
    
    print(f"\nLeverage threshold: {leverage_threshold:.4f}")
    print(f"Number of high-leverage points: {len(high_leverage)}")
    if len(high_leverage) > 0:
        print(f"High-leverage point indices: {high_leverage[:10]}...")  # Show first 10
    
    # Identify influential points (Cook's distance > 4/n)
    cooks_threshold = 4 / n
    influential = np.where(cooks_d > cooks_threshold)[0]
    
    print(f"\nCook's distance threshold: {cooks_threshold:.4f}")
    print(f"Number of influential points: {len(influential)}")
    if len(influential) > 0:
        print(f"Influential point indices: {influential[:10]}...")  # Show first 10
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Leverage values
    axes[0, 0].scatter(range(n), leverage, alpha=0.6, s=30)
    axes[0, 0].axhline(y=leverage_threshold, color='r', linestyle='--', label=f'Threshold: {leverage_threshold:.4f}')
    axes[0, 0].set_xlabel('Observation Index')
    axes[0, 0].set_ylabel('Leverage')
    axes[0, 0].set_title('Leverage Values')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cook's distance
    axes[0, 1].scatter(range(n), cooks_d, alpha=0.6, s=30)
    axes[0, 1].axhline(y=cooks_threshold, color='r', linestyle='--', label=f'Threshold: {cooks_threshold:.4f}')
    axes[0, 1].set_xlabel('Observation Index')
    axes[0, 1].set_ylabel("Cook's Distance")
    axes[0, 1].set_title("Cook's Distance")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Leverage vs Standardized Residuals
    axes[1, 0].scatter(leverage, std_residuals, alpha=0.6, s=30)
    axes[1, 0].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    axes[1, 0].axvline(x=leverage_threshold, color='r', linestyle='--', label='Leverage threshold')
    axes[1, 0].axhline(y=2, color='orange', linestyle='--', label='±2 std')
    axes[1, 0].axhline(y=-2, color='orange', linestyle='--')
    axes[1, 0].set_xlabel('Leverage')
    axes[1, 0].set_ylabel('Standardized Residuals')
    axes[1, 0].set_title('Leverage vs Standardized Residuals')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residuals vs Leverage with Cook's distance contours
    axes[1, 1].scatter(leverage, std_residuals, c=cooks_d, cmap='viridis', alpha=0.6, s=30)
    axes[1, 1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    axes[1, 1].axvline(x=leverage_threshold, color='r', linestyle='--')
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label("Cook's Distance")
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('Standardized Residuals')
    axes[1, 1].set_title("Influence Plot (colored by Cook's Distance)")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved leverage and Cook's distance plot to: {save_path}")
    plt.close()
    
    print("\nInterpretation:")
    print("- High leverage points are outliers in predictor space")
    print("- High Cook's distance indicates influential points that affect the model")
    print("- Points with both high leverage and large residuals are most concerning")
    
    return leverage, cooks_d

# ============================================================================
# Part 14: Multicollinearity Analysis
# ============================================================================

def demonstrate_multicollinearity(n=1000):
    """
    Generate data with multicollinearity and analyze its effects
    
    Parameters:
    -----------
    n : int
        Number of observations
    """
    print("\n" + "=" * 70)
    print("PART 14: MULTICOLLINEARITY ANALYSIS")
    print("=" * 70)
    
    # Generate data with x2 = x1 + 0.9*z
    np.random.seed(42)
    x1 = np.random.randn(n)
    z = np.random.randn(n)
    x2 = x1 + 0.9 * z
    
    # True coefficients
    beta_true = np.array([1.0, 2.0, 3.0])
    
    # Generate response
    epsilon = np.random.randn(n) * 0.5
    y = beta_true[0] + beta_true[1] * x1 + beta_true[2] * x2 + epsilon
    
    # Create design matrix
    X = np.column_stack([np.ones(n), x1, x2])
    
    # (a) Compute condition number
    XtX = X.T @ X
    eigenvalues = np.linalg.eigvalsh(XtX)
    condition_number = eigenvalues[-1] / eigenvalues[0]
    
    print(f"\n(a) Condition Number Analysis:")
    print(f"    Eigenvalues of X^T X: {eigenvalues}")
    print(f"    Condition number: {condition_number:.2f}")
    print(f"    Interpretation: {'High multicollinearity!' if condition_number > 30 else 'Moderate multicollinearity'}")
    
    # (b) Variance inflation
    print(f"\n(b) Variance of Coefficient Estimates:")
    
    # Compute correlation between x1 and x2
    correlation = np.corrcoef(x1, x2)[0, 1]
    print(f"    Correlation between x1 and x2: {correlation:.4f}")
    
    # Estimate coefficients and their variance
    beta_hat = compute_ols_manually(X, y)
    residuals = y - X @ beta_hat
    mse = np.sum(residuals**2) / (n - 3)
    var_beta = mse * np.linalg.inv(XtX)
    
    print(f"\n    Estimated coefficients:")
    print(f"    β0: {beta_hat[0]:.4f} (true: {beta_true[0]})")
    print(f"    β1: {beta_hat[1]:.4f} (true: {beta_true[1]}), std error: {np.sqrt(var_beta[1,1]):.4f}")
    print(f"    β2: {beta_hat[2]:.4f} (true: {beta_true[2]}), std error: {np.sqrt(var_beta[2,2]):.4f}")
    
    # Compute VIF
    from sklearn.linear_model import LinearRegression
    
    # VIF for x1
    model_x1 = LinearRegression()
    model_x1.fit(x2.reshape(-1, 1), x1)
    r2_x1 = model_x1.score(x2.reshape(-1, 1), x1)
    vif_x1 = 1 / (1 - r2_x1)
    
    # VIF for x2
    model_x2 = LinearRegression()
    model_x2.fit(x1.reshape(-1, 1), x2)
    r2_x2 = model_x2.score(x1.reshape(-1, 1), x2)
    vif_x2 = 1 / (1 - r2_x2)
    
    print(f"\n    Variance Inflation Factors (VIF):")
    print(f"    VIF(x1): {vif_x1:.2f}")
    print(f"    VIF(x2): {vif_x2:.2f}")
    print(f"    Rule of thumb: VIF > 10 indicates high multicollinearity")
    
    # (c) Demonstrate with different correlation levels
    print(f"\n(c) Effect of Correlation on Variance:")
    print(f"    {'Correlation':<15} {'Condition #':<15} {'SE(β1)':<15} {'SE(β2)':<15}")
    print(f"    {'-'*60}")
    
    for factor in [0.1, 0.5, 0.9, 0.95, 0.99]:
        z_temp = np.random.randn(n)
        x2_temp = x1 + factor * z_temp
        X_temp = np.column_stack([np.ones(n), x1, x2_temp])
        y_temp = beta_true[0] + beta_true[1] * x1 + beta_true[2] * x2_temp + np.random.randn(n) * 0.5
        
        XtX_temp = X_temp.T @ X_temp
        eigs_temp = np.linalg.eigvalsh(XtX_temp)
        cond_temp = eigs_temp[-1] / eigs_temp[0]
        
        beta_temp = compute_ols_manually(X_temp, y_temp)
        resid_temp = y_temp - X_temp @ beta_temp
        mse_temp = np.sum(resid_temp**2) / (n - 3)
        var_temp = mse_temp * np.linalg.inv(XtX_temp)
        
        corr_temp = np.corrcoef(x1, x2_temp)[0, 1]
        print(f"    {corr_temp:<15.4f} {cond_temp:<15.2f} {np.sqrt(var_temp[1,1]):<15.4f} {np.sqrt(var_temp[2,2]):<15.4f}")
    
    print(f"\n    Explanation:")
    print(f"    - As correlation increases, condition number increases")
    print(f"    - Standard errors of coefficients increase dramatically")
    print(f"    - This causes INSTABILITY (high variance) but NOT BIAS")
    print(f"    - E[β_hat] = β still holds (unbiased)")
    print(f"    - But Var(β_hat) becomes very large")

# ============================================================================
# Main execution
# ============================================================================

def main():
    """
    Main function to execute all parts of Question 1
    """
    print("\n" + "="*70)
    print("QUESTION 1: LINEAR REGRESSION")
    print("="*70)
    
    # Load the dataset
    try:
        # Try to load the dataset
        df = pd.read_csv('linear_regression_dataset.csv')
        print(f"\n✓ Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Separate features and target
        X = df.drop('y', axis=1).values
        y = df['y'].values
        
        print(f"  Features: {df.drop('y', axis=1).columns.tolist()}")
        print(f"  Target: y")
        
    except FileNotFoundError:
        print("\n⚠ Dataset file 'linear_regression_dataset.csv' not found.")
        print("  Generating synthetic data for demonstration...")
        
        # Generate synthetic data
        n_samples = 300
        n_features = 12
        np.random.seed(42)
        
        X = np.random.randn(n_samples, n_features)
        true_coef = np.random.randn(n_features)
        y = X @ true_coef + np.random.randn(n_samples) * 2
        
        print(f"✓ Generated synthetic dataset: {n_samples} samples, {n_features} features")
    
    # Part 7: Compare manual OLS with sklearn
    manual_beta, sklearn_beta = compare_with_sklearn(X, y)
    
    # Compute predictions for analysis
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    y_pred = X_with_intercept @ manual_beta
    residuals = y - y_pred
    
    # Part 8: Plot residuals vs fitted values
    plot_residuals_vs_fitted(y, y_pred)
    
    # Part 9: Q-Q plot
    plot_qq_plot(residuals)
    
    # Part 10: Analyze assumptions
    analyze_assumptions(residuals, y_pred)
    
    # Part 11: Leverage and Cook's distance
    leverage, cooks_d = compute_leverage_and_cooks_distance(X_with_intercept, y)
    
    # Part 14: Multicollinearity demonstration
    demonstrate_multicollinearity()
    
    print("\n" + "="*70)
    print("QUESTION 1 COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - residuals_vs_fitted.png")
    print("  - qq_plot.png")
    print("  - leverage_cooks.png")
    print("\nAll theoretical derivations are provided in the LaTeX report.")

if __name__ == "__main__":
    main()
