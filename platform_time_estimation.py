"""
Platform Time Contribution Estimation using Ordinal Regression

This script solves the ecological inference problem where:
- We know total online time (categorical: 0-2h, 2-4h, 4+h)
- We know which platforms users use
- We want to estimate average time contribution per platform

Approach: Ordinal Logistic Regression (Proportional Odds Model)
- Respects the ordinal nature of time categories
- Models cumulative probabilities for each category
- Coefficients represent log-odds of being in higher time categories

The coefficients βᵢ represent the contribution of each platform to higher time usage.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try importing mord (ordinal regression library)
try:
    import mord
    ORDINAL_AVAILABLE = True
except ImportError:
    ORDINAL_AVAILABLE = False
    print("Warning: mord library not available. Install with: pip install mord")
    print("Falling back to statsmodels ordinal regression")

# Try statsmodels as alternative
try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels ordinal model not available.")


def encode_hours_category_ordinal(category):
    """
    Convert categorical time ranges to ordinal codes for ordinal regression

    Less than 2 hours -> 0
    2-4 hours -> 1
    more than 4 hours -> 2
    Others/NaN -> NaN
    """
    if pd.isna(category):
        return np.nan

    category_str = str(category).strip().lower()

    if 'less than 2' in category_str or category_str == '0-2':
        return 0
    elif '2-4' in category_str or category_str == '2-4 hours':
        return 1
    elif 'more than 4' in category_str or '4+' in category_str or category_str == 'more than 4 hours':
        return 2
    else:
        return np.nan


def encode_hours_category_midpoint(category):
    """
    Convert categorical time ranges to numeric midpoints (in hours)
    Used for interpretation and visualization

    Less than 2 hours -> 1.0
    2-4 hours -> 3.0
    more than 4 hours -> 5.0
    Others/NaN -> NaN
    """
    if pd.isna(category):
        return np.nan

    category_str = str(category).strip().lower()

    if 'less than 2' in category_str or category_str == '0-2':
        return 1.0
    elif '2-4' in category_str or category_str == '2-4 hours':
        return 3.0
    elif 'more than 4' in category_str or '4+' in category_str or category_str == 'more than 4 hours':
        return 5.0
    else:
        return np.nan


def prepare_regression_data(df):
    """
    Prepare data for both ordinal and OLS regression analysis

    Returns:
        X: Binary indicators for each platform (n_samples x n_platforms)
        y_ordinal: Ordinal encoded time (0, 1, 2)
        y_midpoint: Midpoint encoded time (1.0, 3.0, 5.0)
        platform_names: List of platform names
        valid_mask: Boolean mask for valid samples
    """
    # Platform columns
    platform_cols = {
        'Q10_a': 'Facebook',
        'Q10_b': 'Youtube',
        'Q10_c': 'TikTok',
        'Q10_d': 'Viber',
        'Q10_e': 'Instagram',
        'Q10_f': 'Messenger',
        'Q10_g': 'Others'
    }

    # Encode time categories for both methods
    df['Q28_ordinal'] = df['Q28'].apply(encode_hours_category_ordinal)
    df['Q28_midpoint'] = df['Q28'].apply(encode_hours_category_midpoint)

    # Create feature matrix: binary indicators for platform usage
    X_cols = []
    platform_names = []

    for col, name in platform_cols.items():
        if col in df.columns:
            # Convert to binary (1 if uses platform, 0 otherwise)
            df[f'{col}_binary'] = df[col].fillna(0).apply(lambda x: 1 if x == 1.0 else 0)
            X_cols.append(f'{col}_binary')
            platform_names.append(name)

    # Create feature matrix and targets
    X = df[X_cols].values
    y_ordinal = df['Q28_ordinal'].values
    y_midpoint = df['Q28_midpoint'].values

    # Valid samples: have time data and use at least one platform
    valid_mask = (~np.isnan(y_ordinal)) & (X.sum(axis=1) > 0)

    return X[valid_mask], y_ordinal[valid_mask], y_midpoint[valid_mask], platform_names, valid_mask, X_cols


def fit_ols_regression(X, y_midpoint, platform_names):
    """
    Fit OLS regression with midpoint coding to estimate platform time contributions

    Returns:
        results_df: DataFrame with platform contributions and statistics
        model: Fitted regression model
    """
    from sklearn.linear_model import LinearRegression

    # Fit linear regression
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y_midpoint)

    # Get coefficients
    coefficients = model.coef_
    intercept = model.intercept_

    # Calculate R-squared
    y_pred = model.predict(X)
    ss_res = np.sum((y_midpoint - y_pred) ** 2)
    ss_tot = np.sum((y_midpoint - np.mean(y_midpoint)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Create results dataframe
    results_df = pd.DataFrame({
        'Platform': platform_names,
        'Time_Contribution_Hours': coefficients,
        'Percentage_of_Total': (coefficients / coefficients.sum() * 100) if coefficients.sum() > 0 else [0] * len(coefficients)
    })

    # Sort by contribution
    results_df = results_df.sort_values('Time_Contribution_Hours', ascending=False)

    # Add model statistics
    results_df['R_Squared'] = r_squared
    results_df['Intercept'] = intercept
    results_df['Mean_Total_Time'] = np.mean(y_midpoint)
    results_df['Method'] = 'OLS_Midpoint'

    return results_df, model


def fit_ordinal_regression(X, y_ordinal, platform_names):
    """
    Fit ordinal logistic regression (proportional odds model)

    Returns:
        results_df: DataFrame with platform contributions (coefficients) and statistics
        model: Fitted ordinal regression model
    """
    if STATSMODELS_AVAILABLE:
        # Use statsmodels OrderedModel
        import pandas as pd
        X_df = pd.DataFrame(X, columns=platform_names)

        try:
            model = OrderedModel(y_ordinal, X_df, distr='logit')
            result = model.fit(method='bfgs', disp=False)

            coefficients = result.params[:-2]  # Exclude thresholds
            std_errors = result.bse[:-2]  # Standard errors (exclude thresholds)

            # Calculate 95% CI
            ci_lower = coefficients - 1.96 * std_errors
            ci_upper = coefficients + 1.96 * std_errors

            # Create results dataframe
            results_df = pd.DataFrame({
                'Platform': platform_names,
                'Ordinal_Coefficient': coefficients.values,
                'Std_Error': std_errors.values,
                'CI_Lower': ci_lower.values,
                'CI_Upper': ci_upper.values,
                'P_Value': result.pvalues[:-2].values
            })

            # Sort by coefficient magnitude
            results_df = results_df.sort_values('Ordinal_Coefficient', ascending=False)
            results_df['Method'] = 'Ordinal_Logistic'
            results_df['Pseudo_R_Squared'] = result.prsquared

            return results_df, result

        except Exception as e:
            print(f"   Error fitting statsmodels ordinal model: {e}")
            print("   Using fallback OLS on ordinal codes...")
            return fit_fallback_ordinal(X, y_ordinal, platform_names)

    elif ORDINAL_AVAILABLE:
        # Use mord library
        try:
            model = mord.LogisticAT(alpha=0)  # No regularization
            model.fit(X, y_ordinal.astype(int))

            coefficients = model.coef_

            results_df = pd.DataFrame({
                'Platform': platform_names,
                'Ordinal_Coefficient': coefficients,
            })

            results_df = results_df.sort_values('Ordinal_Coefficient', ascending=False)
            results_df['Method'] = 'Ordinal_Logistic_MORD'

            return results_df, model

        except Exception as e:
            print(f"   Error fitting mord ordinal model: {e}")
            print("   Using fallback OLS on ordinal codes...")
            return fit_fallback_ordinal(X, y_ordinal, platform_names)

    else:
        print("   No ordinal regression library available. Using fallback OLS on ordinal codes...")
        return fit_fallback_ordinal(X, y_ordinal, platform_names)


def fit_fallback_ordinal(X, y_ordinal, platform_names):
    """
    Fallback method: OLS on ordinal codes (0, 1, 2)
    """
    from sklearn.linear_model import LinearRegression

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y_ordinal)

    coefficients = model.coef_

    # Calculate R-squared
    y_pred = model.predict(X)
    ss_res = np.sum((y_ordinal - y_pred) ** 2)
    ss_tot = np.sum((y_ordinal - np.mean(y_ordinal)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Approximate standard errors for OLS
    n = len(y_ordinal)
    p = len(coefficients)
    mse = ss_res / (n - p - 1)

    # Calculate standard errors (approximate)
    var_coef = mse * np.linalg.inv(X.T @ X).diagonal()
    std_errors = np.sqrt(var_coef)

    # 95% CI
    ci_lower = coefficients - 1.96 * std_errors
    ci_upper = coefficients + 1.96 * std_errors

    results_df = pd.DataFrame({
        'Platform': platform_names,
        'Ordinal_Coefficient': coefficients,
        'Std_Error': std_errors,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper
    })

    results_df = results_df.sort_values('Ordinal_Coefficient', ascending=False)
    results_df['Method'] = 'OLS_on_Ordinal_Codes'
    results_df['R_Squared'] = r_squared

    return results_df, model


def calculate_platform_time_distribution(df, results_df):
    """
    Calculate the distribution of users across time categories for each platform

    Returns:
        distribution_df: DataFrame with percentage of users in each time category per platform
    """
    platform_cols = {
        'Q10_a': 'Facebook',
        'Q10_b': 'Youtube',
        'Q10_c': 'TikTok',
        'Q10_d': 'Viber',
        'Q10_e': 'Instagram',
        'Q10_f': 'Messenger',
        'Q10_g': 'Others'
    }

    distribution_data = []

    for col, name in platform_cols.items():
        if col in df.columns:
            # Get users who use this platform
            platform_users = df[df[col] == 1.0].copy()

            if len(platform_users) > 0:
                # Count users in each time category
                time_counts = platform_users['Q28'].value_counts()
                total_users = len(platform_users)

                # Calculate percentages
                less_than_2 = (time_counts.get('Less than 2 hours', 0) / total_users * 100)
                two_to_four = (time_counts.get('2-4 hours', 0) / total_users * 100)
                more_than_4 = (time_counts.get('more than 4 hours', 0) / total_users * 100)

                distribution_data.append({
                    'Platform': name,
                    'Less_than_2_hours_%': less_than_2,
                    '2_to_4_hours_%': two_to_four,
                    'More_than_4_hours_%': more_than_4,
                    'Total_Users': total_users
                })

    distribution_df = pd.DataFrame(distribution_data)

    # Merge with regression results
    if not results_df.empty and not distribution_df.empty:
        distribution_df = distribution_df.merge(
            results_df[['Platform', 'Time_Contribution_Hours']],
            on='Platform',
            how='left'
        )

        # Sort by time contribution
        distribution_df = distribution_df.sort_values('Time_Contribution_Hours', ascending=False)

    return distribution_df


def main():
    """Main execution function - runs both ordinal and OLS regression"""
    print("="*70)
    print("Platform Time Contribution Estimation")
    print("DUAL APPROACH: Ordinal Regression + OLS with Midpoint Coding")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv('clean/CLEAN_FY26.csv')
    print(f"   Total respondents: {len(df)}")

    # Prepare data
    print("\n2. Preparing regression data...")
    X, y_ordinal, y_midpoint, platform_names, valid_mask, X_cols = prepare_regression_data(df)
    print(f"   Valid samples for regression: {len(y_ordinal)}")
    print(f"   Platforms analyzed: {', '.join(platform_names)}")
    print(f"   Time categories: 0=<2h, 1=2-4h, 2=>4h")

    # Fit ORDINAL regression
    print("\n3. Fitting ORDINAL regression model...")
    ordinal_results_df, ordinal_model = fit_ordinal_regression(X, y_ordinal, platform_names)

    print(f"   Method: {ordinal_results_df['Method'].iloc[0]}")
    if 'Pseudo_R_Squared' in ordinal_results_df.columns:
        print(f"   Pseudo R-squared: {ordinal_results_df['Pseudo_R_Squared'].iloc[0]:.4f}")
    elif 'R_Squared' in ordinal_results_df.columns:
        print(f"   R-squared: {ordinal_results_df['R_Squared'].iloc[0]:.4f}")

    print("\n   Ordinal Coefficients (log-odds scale):")
    print("   " + "-"*66)
    for idx, row in ordinal_results_df.iterrows():
        coef_col = 'Ordinal_Coefficient'
        print(f"   {row['Platform']:15} | {row[coef_col]:7.4f}")

    # Fit OLS regression
    print("\n4. Fitting OLS regression model (midpoint coding)...")
    ols_results_df, ols_model = fit_ols_regression(X, y_midpoint, platform_names)

    print(f"   Method: {ols_results_df['Method'].iloc[0]}")
    print(f"   R-squared: {ols_results_df['R_Squared'].iloc[0]:.4f}")
    print(f"   Intercept: {ols_results_df['Intercept'].iloc[0]:.4f} hours")
    print(f"   Mean total time: {ols_results_df['Mean_Total_Time'].iloc[0]:.4f} hours")

    print("\n   OLS Platform Time Contributions:")
    print("   " + "-"*66)
    for idx, row in ols_results_df.iterrows():
        print(f"   {row['Platform']:15} | {row['Time_Contribution_Hours']:6.3f} hours | {row['Percentage_of_Total']:6.2f}%")

    # Calculate distribution
    print("\n5. Calculating time distribution by platform...")
    distribution_df = calculate_platform_time_distribution(df, ols_results_df)

    print("\n   Observed Time Distribution by Platform:")
    print("   " + "-"*66)
    for idx, row in distribution_df.iterrows():
        print(f"   {row['Platform']:15} | <2h: {row['Less_than_2_hours_%']:5.1f}% | 2-4h: {row['2_to_4_hours_%']:5.1f}% | 4+h: {row['More_than_4_hours_%']:5.1f}%")

    # Save results
    print("\n6. Saving results...")

    # Save OLS regression coefficients
    ols_output_file = 'clean/platform_time_contributions_ols.csv'
    ols_results_df.to_csv(ols_output_file, index=False)
    print(f"   OLS results saved to: {ols_output_file}")

    # Save ordinal regression coefficients
    ordinal_output_file = 'clean/platform_time_contributions_ordinal.csv'
    ordinal_results_df.to_csv(ordinal_output_file, index=False)
    print(f"   Ordinal results saved to: {ordinal_output_file}")

    # Save distribution for visualization
    distribution_file = 'clean/platform_time_distribution.csv'
    distribution_df.to_csv(distribution_file, index=False)
    print(f"   Distribution data saved to: {distribution_file}")

    print("\n" + "="*70)
    print("COMPLETE - Results saved to clean/ directory")
    print("="*70)

    return ols_results_df, ordinal_results_df


if __name__ == "__main__":
    ols_results_df, ordinal_results_df = main()
