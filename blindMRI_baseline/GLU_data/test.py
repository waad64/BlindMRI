import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Load the glucose dataset
df = pd.read_csv('outputs/glucose_data.csv')
glu = df['GLU']

# Summary stats
print("📈 Summary Statistics:")
print(glu.describe())

# Normality tests
shapiro_stat, shapiro_p = stats.shapiro(glu)
ad_result = stats.anderson(glu)
ks_stat, ks_p = stats.kstest(glu, 'norm', args=(glu.mean(), glu.std()))

# Plot setup
plt.figure(figsize=(16, 10))

# 1. Histogram + KDE
plt.subplot(2, 2, 1)
sns.histplot(glu, bins=30, kde=True, color='lightcoral')
plt.title("Glucose Level Distribution (Histogram + KDE)")
plt.xlabel("Blood Glucose Level (mg/dL)")
plt.ylabel("Frequency")

# 2. Boxplot
plt.subplot(2, 2, 2)
sns.boxplot(x=glu, color='salmon')
plt.title("Glucose Level Boxplot")
plt.xlabel("Blood Glucose Level (mg/dL)")

# 3. Q-Q Plot
plt.subplot(2, 2, 3)
sm.qqplot(glu, line='s', ax=plt.gca())
plt.title("Q-Q Plot of Glucose Levels")

# 4. Normality Test Summary
plt.subplot(2, 2, 4)
plt.axis('off')
test_summary = f"""
🧪 Normality Test Results for Glucose Levels:

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
