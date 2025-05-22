import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('outputs/raw_dataset_with_map.csv')
map_data = df['MAP']

# Plot setup
plt.figure(figsize=(16, 12))

# 1. Histogram + KDE
plt.subplot(2, 2, 1)
sns.histplot(map_data, bins=30, kde=True, color='teal')
plt.title('MAP Distribution (Histogram + KDE)')
plt.xlabel('MAP')
plt.ylabel('Frequency')

# 2. KDE only (for clarity)
plt.subplot(2, 2, 2)
sns.kdeplot(map_data, color='blue', fill=True)
plt.title('KDE of MAP')
plt.xlabel('MAP')
plt.ylabel('Density')

# 3. Boxplot
plt.subplot(2, 2, 3)
sns.boxplot(x=map_data, color='lightgreen')
plt.title('Boxplot of MAP')
plt.xlabel('MAP')

# 4. Q-Q Plot
plt.subplot(2, 2, 4)
sm.qqplot(map_data, line='s', ax=plt.gca())
plt.title('Q-Q Plot of MAP')

plt.tight_layout()
plt.show()

# Normality tests

# Anderson-Darling test
ad_result = stats.anderson(map_data)

# Kolmogorov-Smirnov test against normal distribution
ks_stat, ks_p = stats.kstest(map_data, 'norm', args=(map_data.mean(), map_data.std()))

# Print test summaries
print("🧪 Normality Test Results for MAP:")
print(f"• Anderson-Darling Test Statistic: {ad_result.statistic:.4f}")
print(f"  Critical Values: {ad_result.critical_values}")
print(f"  Significance Levels: {ad_result.significance_level}")
print(f"• Kolmogorov-Smirnov Test Statistic: {ks_stat:.4f}")
print(f"  p-value: {ks_p:.4f}")
