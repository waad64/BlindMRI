import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

# Load the data
df = pd.read_csv('outputs/blood_pressure_data.csv')

# Define a function to analyze and plot
def analyze_bp(data, label):
    plt.figure(figsize=(16, 10))
    
    # 1. Histogram + KDE
    plt.subplot(2, 2, 1)
    sns.histplot(data, kde=True, color='skyblue', bins=20)
    plt.title(f'{label} - Histogram & KDE')
    plt.xlabel(f'{label} BP')

    # 2. Boxplot
    plt.subplot(2, 2, 2)
    sns.boxplot(x=data, color='lightgreen')
    plt.title(f'{label} - Boxplot')

    # 3. Q-Q Plot
    plt.subplot(2, 2, 3)
    sm.qqplot(data, line='s', ax=plt.gca())
    plt.title(f'{label} - Q-Q Plot')

    # 4. Normality Tests
    shapiro_stat, shapiro_p = stats.shapiro(data)
    ad_result = stats.anderson(data)
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))

    plt.subplot(2, 2, 4)
    plt.axis('off')
    test_summary = f'''
    Normality Tests for {label} BP:
    - Shapiro-Wilk p-value: {shapiro_p:.4f}
    - Anderson-Darling stat: {ad_result.statistic:.4f}
      Critical values: {ad_result.critical_values}
    - Kolmogorov-Smirnov p-value: {ks_p:.4f}
    '''
    plt.text(0.01, 0.5, test_summary, fontsize=12, verticalalignment='center')

    plt.tight_layout()
    plt.show()

# Analyze Systolic and Diastolic
analyze_bp(df['SBP'], 'Systolic')
analyze_bp(df['DBP'], 'Diastolic')
