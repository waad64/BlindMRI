# Analysis of Anxiety_Level Distribution

This analysis examines the distribution of the `Anxiety_Level` variable using statistical plots and normality tests.

## Visualizations

### Histogram + KDE of Anxiety_Level (Figure 1)

The histogram with Kernel Density Estimate (KDE) clearly shows a bimodal distribution for `Anxiety_Level`. There are two distinct peaks, one near 0.0 and another near 1.0, with very few observations in between. This suggests that the `Anxiety_Level` data is concentrated at the extreme ends of its range.

### Boxplot of Anxiety_Level (Figure 2)

The boxplot confirms the bimodal nature of the data. It appears as a single, wide box spanning almost the entire range from 0.0 to 1.0, with the median line likely obscured by the concentration of data at the extremes. The absence of visible whiskers or outliers reinforces the idea that the data is not normally distributed but rather clustered at the minimum and maximum values.

### Q-Q Plot of Anxiety_Level (Figure 3)

The Q-Q plot strikingly deviates from the theoretical quantiles (the red line), particularly showing horizontal segments at approximately 0 and 1 on the Sample Quantiles axis. This step-like pattern is highly indicative of a variable that primarily takes on two distinct values, rather than a continuous or normally distributed variable. The significant departure from the red line confirms that the data is not normally distributed.

## Normality Tests

### Kolmogorov-Smirnov Test

* **Statistic:** 0.3413
* **p-value:** 0.0000

The very low p-value (0.0000) is significantly less than typical significance levels (e.g., 0.05). This leads to the rejection of the null hypothesis ($H_0$), which states that the data comes from a normal distribution. Therefore, the Kolmogorov-Smirnov test indicates that the `Anxiety_Level` data is **not normally distributed**.

### Anderson-Darling Test

* **Statistic:** 1432.2957

Comparing the calculated statistic to the critical values at various significance levels:

* **Significance Level 15.0%:** Critical Value = 0.576 --> **Reject $H_0$** (1432.2957 > 0.576)
* **Significance Level 10.0%:** Critical Value = 0.656 --> **Reject $H_0$** (1432.2957 > 0.656)
* **Significance Level 5.0%:** Critical Value = 0.787 --> **Reject $H_0$** (1432.2957 > 0.787)
* **Significance Level 2.5%:** Critical Value = 0.918 --> **Reject $H_0$** (1432.2957 > 0.918)
* **Significance Level 1.0%:** Critical Value = 1.091 --> **Reject $H_0$** (1432.2957 > 1.091)

In all tested significance levels, the Anderson-Darling test statistic is substantially larger than the critical values. This consistently leads to the rejection of the null hypothesis ($H_0$), which states that the data comes from a normal distribution. Therefore, the Anderson-Darling test also strongly indicates that the `Anxiety_Level` data is **not normally distributed**.

## Conclusion

All visual and statistical analyses consistently demonstrate that the `Anxiety_Level` variable is **not normally distributed**. The data exhibits a clear bimodal pattern, with values concentrated at the extremes (around 0 and 1), rather than following a bell-shaped curve. This non-normal distribution should be considered when applying statistical methods that assume normality.