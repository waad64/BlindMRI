import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('outputs/EDA_data.csv')

eda_signal = df['EDA']

# 📈 Summary statistics
print("📊 Summary Statistics:")
print(eda_signal.describe())

# 🧪 Normality Tests
shapiro_stat, shapiro_p = stats.shapiro(eda_signal)
ad_result = stats.anderson(eda_signal)
ks_stat, ks_p = stats.kstest(eda_signal, 'norm', args=(eda_signal.mean(), eda_signal.std()))

# 📉 Plotting
plt.figure(figsize=(16, 10))

# 1. Histogram + KDE
plt.subplot(2, 2, 1)
sns.histplot(eda_signal, kde=True, color='skyblue', bins=30)
plt.title("EDA Signal Distribution (Histogram + KDE)")
plt.xlabel("EDA Value")
plt.ylabel("Frequency")

# 2. Boxplot
plt.subplot(2, 2, 2)
sns.boxplot(x=eda_signal, color='lightgreen')
plt.title("EDA Signal Boxplot")
plt.xlabel("EDA Value")

# 3. Q-Q Plot
plt.subplot(2, 2, 3)
sm.qqplot(eda_signal, line='s', ax=plt.gca())
plt.title("Q-Q Plot of EDA Signal")

# 4. Test Results Display
plt.subplot(2, 2, 4)
plt.axis('off')
test_summary = f"""
🧪 Normality Test Results for EDA:

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
