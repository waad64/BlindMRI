import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Load the heart rate dataset
df = pd.read_csv('outputs/heart_rate_data.csv')
hr = df['HR']

# Summary statistics
print("📈 Summary Statistics:")
print(hr.describe())

# Normality tests
shapiro_stat, shapiro_p = stats.shapiro(hr)
ad_result = stats.anderson(hr)
ks_stat, ks_p = stats.kstest(hr, 'norm', args=(hr.mean(), hr.std()))

# Plot setup
plt.figure(figsize=(16, 10))

# 1. Histogram + KDE
plt.subplot(2, 2, 1)
sns.histplot(hr, bins=100, kde=True, color='steelblue')
plt.title("Heart Rate Distribution (Histogram + KDE)")
plt.xlabel("Heart Rate (bpm)")
plt.ylabel("Frequency")

# 2. Boxplot
plt.subplot(2, 2, 2)
sns.boxplot(x=hr, color='skyblue')
plt.title("Heart Rate Boxplot")
plt.xlabel("Heart Rate (bpm)")

# 3. Q-Q Plot
plt.subplot(2, 2, 3)
sm.qqplot(hr, line='s', ax=plt.gca())
plt.title("Q-Q Plot of Heart Rate")

# 4. Normality Test Summary
plt.subplot(2, 2, 4)
plt.axis('off')
test_summary = f"""
🧪 Normality Test Results for Heart Rate:

• Shapiro-Wilk Test:
  - Statistic = {shapiro_stat:.4f}
  - p-value   = {shapiro_p:.4f}

• Anderson-Darling Test:
  - Statistic       = {ad_result.statistic:.4f}
  - Critical Values = {ad_result.critical_values}

• Kolmogorov-Smirnov Test:
  - Statistic = {ks_stat:.4f}
  - p-value   = {ks_p:.4f}
"""
plt.text(0.01, 0.5, test_summary, fontsize=11, verticalalignment='center', family='monospace')

plt.tight_layout()
plt.show()
