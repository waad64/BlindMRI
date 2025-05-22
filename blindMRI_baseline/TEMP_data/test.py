import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load the temperature dataset
try:
    df = pd.read_csv('outputs/TEMP_data.csv')
except FileNotFoundError:
    print("Error: 'outputs/TEMP_data.csv' not found. Please ensure the file path is correct.")
    exit()

# Ensure the 'TEMP' column exists (keeping your original case sensitivity)
if 'TEMP' not in df.columns:
    print("Error: 'TEMP' column not found in the dataset. Please check the column name, ensuring it matches 'TEMP' exactly.")
    exit()

# --- 1. Summary Statistics ---
print("📈 Summary Statistics for TEMP Signal:")
print(df['TEMP'].describe())
print("\n" + "="*50 + "\n")

# --- 2. Visualizations ---
plt.figure(figsize=(15, 12))

# 2.1 Histogram with KDE
plt.subplot(2, 2, 1)
sns.histplot(df['TEMP'], bins=30, kde=True, color='skyblue')
plt.title("TEMP Signal Distribution (Histogram + KDE)")
plt.xlabel("TEMP Value")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.7)

# 2.2 KDE Plot
plt.subplot(2, 2, 2)
sns.kdeplot(df['TEMP'], fill=True, color='blue')
plt.title("KDE of TEMP Signal")
plt.xlabel("TEMP Value")
plt.ylabel("Density")
plt.grid(True, linestyle='--', alpha=0.7)

# 2.3 Boxplot
plt.subplot(2, 2, 3)
sns.boxplot(y=df['TEMP'], color='lightgreen')
plt.title("Boxplot of TEMP Signal")
plt.ylabel("TEMP Value")
plt.grid(True, linestyle='--', alpha=0.7)

# 2.4 Q-Q Plot
plt.subplot(2, 2, 4)
stats.probplot(df['TEMP'], dist="norm", plot=plt)
plt.title("Q-Q Plot of TEMP Signal")
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# --- 3. Normality Tests ---
print("🧪 Normality Test Results for TEMP Signal:")

# Shapiro-Wilk Test
shapiro_statistic, shapiro_p_value = stats.shapiro(df['TEMP'])
print(f"• Shapiro-Wilk Test:")
print(f"  - Statistic: {shapiro_statistic:.4f}")
print(f"  - p-value: {shapiro_p_value:.4f}")
if shapiro_p_value < 0.05:
    print("  - Interpretation: Reject the null hypothesis (data is NOT normal)")
else:
    print("  - Interpretation: Fail to reject the null hypothesis (data appears normal)")

# Kolmogorov-Smirnov Test (using kstest against a normal distribution with estimated parameters)
# Note: For accurate K-S against a normal distribution with estimated mean/std,
# the Lilliefors test is more appropriate. However, if lilliefors is not available,
# kstest with estimated parameters is the next best common alternative.
# It's important to be aware that this might slightly inflate Type I error rates.
mean_temp_for_ks = df['TEMP'].mean()
std_temp_for_ks = df['TEMP'].std()
ks_statistic, ks_p_value = stats.kstest(df['TEMP'], 'norm', args=(mean_temp_for_ks, std_temp_for_ks))
print(f"\n• Kolmogorov-Smirnov Test (against normal with estimated parameters):")
print(f"  - Statistic: {ks_statistic:.4f}")
print(f"  - p-value: {ks_p_value:.4f}")
if ks_p_value < 0.05:
    print("  - Interpretation: Reject the null hypothesis (data is NOT normal)")
else:
    print("  - Interpretation: Fail to reject the null hypothesis (data appears normal)")


# --- 4. Outlier Detection (Z-score method) ---
print("\n🔍 Outlier Detection (|z| > 3):")
mean_temp_outlier = df['TEMP'].mean()
std_temp_outlier = df['TEMP'].std()
df['z_score_TEMP'] = (df['TEMP'] - mean_temp_outlier) / std_temp_outlier
outliers_zscore = df[(np.abs(df['z_score_TEMP']) > 3)]
print(f"Detected {len(outliers_zscore)} outliers (|z| > 3) out of {len(df)} values.")
if not outliers_zscore.empty:
    print("Example outliers (first 5):")
    # Corrected to use 'TEMP' column name for display consistency
    print(outliers_zscore[['TEMP', 'z_score_TEMP']].head())
else:
    print("No significant outliers detected using Z-score method (|z| > 3).")

print("\n" + "="*50 + "\n")