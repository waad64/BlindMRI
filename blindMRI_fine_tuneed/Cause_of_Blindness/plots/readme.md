# Analysis of Cause_Blindness Distribution

This README.md file presents an analysis of the distribution of the `Cause_Blindness` variable, utilizing statistical visualizations and normality test results.

## Visualizations

### Histogram + KDE of Cause_Blindness (Figure 1)

The histogram with Kernel Density Estimate (KDE) for `Cause_Blindness` reveals a distinct **multimodal distribution**, specifically with three prominent peaks. These peaks are located approximately at values 1, 2, and 3, indicating that the `Cause_Blindness` variable is likely categorical or ordinal, taking on a limited number of discrete values. The high frequency at these specific points, with very low frequencies in between, strongly suggests that the data represents distinct categories rather than a continuous variable. The KDE curves are centered around these peaks, reinforcing the discrete nature of the data.

### Boxplot of Cause_Blindness (Figure 2)

The boxplot of `Cause_Blindness` further supports the interpretation of a discrete variable. Instead of a typical box and whisker representation of a continuous range, the boxplot here appears to encompass the range of the observed discrete values. The "box" spans from approximately 1 to 2, and a "whisker" extends to 3, which aligns with the peaks observed in the histogram. The visual representation here is unusual for a continuous variable and reinforces the idea that `Cause_Blindness` represents distinct categories.

### Q-Q Plot of Cause_Blindness (Figure 3)

The Q-Q plot for `Cause_Blindness` displays a pattern highly characteristic of **discrete, non-normally distributed data**. The points form distinct horizontal segments at approximately sample quantiles 1, 2, and 3. This "stair-step" or "plateau" appearance is a strong indicator that the variable takes on a limited number of specific values rather than being continuous and normally distributed. The significant deviation from the red theoretical normal distribution line confirms that the data is not normal.

## Normality Tests

Two sets of Kolmogorov-Smirnov and Anderson-Darling test results were provided. We will analyze both.

### Set 1 Test Results:

📊 **Kolmogorov-Smirnov Test:**
* **Statistic:** 0.2188
* **p-value:** 0.0000

The p-value of 0.0000 is extremely small, significantly less than any common significance level (e.g., 0.05). This leads to a strong **rejection of the null hypothesis** ($H_0$), which states that the data comes from a normal distribution.

📈 **Anderson-Darling Test:**
* **Statistic:** 462.9631
* **Significance Level 15.0%:** Critical Value = 0.576 --> **Reject H0** (462.9631 >> 0.576)
* **Significance Level 10.0%:** Critical Value = 0.656 --> **Reject H0** (462.9631 >> 0.656)
* **Significance Level 5.0%:** Critical Value = 0.787 --> **Reject H0** (462.9631 >> 0.787)
* **Significance Level 2.5%:** Critical Value = 0.918 --> **Reject H0** (462.9631 >> 0.918)
* **Significance Level 1.0%:** Critical Value = 1.091 --> **Reject H0** (462.9631 >> 1.091)

In all tested significance levels, the Anderson-Darling test statistic (462.9631) is substantially greater than the critical values. This consistently leads to the **rejection of the null hypothesis** ($H_0$), indicating non-normality.

### Set 2 Test Results:

📊 **Kolmogorov-Smirnov Test:**
* **Statistic:** 0.2585
* **p-value:** 0.0000

Again, the p-value of 0.0000 is extremely small, leading to a strong **rejection of the null hypothesis** ($H_0$).

📈 **Anderson-Darling Test:**
* **Statistic:** 673.1068
* **Significance Level 15.0%:** Critical Value = 0.576 --> **Reject H0** (673.1068 >> 0.576)
* **Significance Level 10.0%:** Critical Value = 0.656 --> **Reject H0** (673.1068 >> 0.656)
* **Significance Level 5.0%:** Critical Value = 0.787 --> **Reject H0** (673.1068 >> 0.787)
* **Significance Level 2.5%:** Critical Value = 0.918 --> **Reject H0** (673.1068 >> 0.918)
* **Significance Level 1.0%:** Critical Value = 1.091 --> **Reject H0** (673.1068 >> 1.091)

Similar to the first set of results, the Anderson-Darling test statistic (673.1068) is overwhelmingly larger than all critical values, leading to a consistent **rejection of the null hypothesis** ($H_0$).

## Conclusion

Both the visual analyses and the statistical normality tests (Kolmogorov-Smirnov and Anderson-Darling from two separate runs) strongly and consistently indicate that the `Cause_Blindness` variable **is not normally distributed**. The visualizations, particularly the histogram and Q-Q plot, clearly show that this variable is **discrete and multimodal**, taking on specific values (likely 1, 2, and 3) rather than being continuous and following a bell-shaped curve. 