# Analysis of Blindness_Duration Distribution

This README.md file provides an analysis of the distribution of the `Blindness_Duration` variable, incorporating statistical visualizations and normality test results.

## Visualizations

### Histogram + KDE of Blindness_Duration (Figure 1)

The histogram with Kernel Density Estimate (KDE) for `Blindness_Duration` shows a highly **right-skewed distribution**. There's a prominent peak in frequency at the lower end of the duration (around 0-25), indicating that many instances of blindness have a shorter duration. The frequency gradually decreases as the duration increases, with a long tail extending towards higher values. The KDE curve visually confirms this skewness, peaking sharply on the left and tapering off to the right.

### Boxplot of Blindness_Duration (Figure 2)

The boxplot of `Blindness_Duration` further illustrates the right-skewed nature of the data. The median line within the box is positioned significantly towards the left side of the box, closer to the first quartile. The right whisker is considerably longer than the left whisker, and there appear to be numerous outliers on the higher end of the duration, indicating observations with exceptionally long blindness durations. This confirms the non-symmetrical distribution and the presence of larger values stretching the upper tail.

### Q-Q Plot of Blindness_Duration (Figure 3)

The Q-Q plot clearly indicates a **departure from normality**. The observed data points (blue circles) do not follow the theoretical normal distribution line (red line). Specifically, the points initially lie below the line, then cross it and lie above it, forming an 'S' shape. This pattern, where the tails of the distribution diverge from the theoretical line, is characteristic of a skewed distribution. In this case, the upward curve at the higher quantiles suggests a heavier right tail than a normal distribution, consistent with the right-skewness observed in the histogram and boxplot.

## Normality Tests

### Kolmogorov-Smirnov Test

* **Statistic:** 0.2188
* **p-value:** 0.0000

The p-value of 0.0000 is extremely small (much less than typical significance levels like 0.05). This strong evidence leads to the **rejection of the null hypothesis** ($H_0$), which states that the data is drawn from a normal distribution. Therefore, the Kolmogorov-Smirnov test concludes that `Blindness_Duration` is **not normally distributed**.

### Anderson-Darling Test

* **Statistic:** 462.9631

Comparing the calculated statistic to the critical values at various significance levels:

* **Significance Level 15.0%:** Critical Value = 0.576 --> **Reject $H_0$** (462.9631 >> 0.576)
* **Significance Level 10.0%:** Critical Value = 0.656 --> **Reject $H_0$** (462.9631 >> 0.656)
* **Significance Level 5.0%:** Critical Value = 0.787 --> **Reject $H_0$** (462.9631 >> 0.787)
* **Significance Level 2.5%:** Critical Value = 0.918 --> **Reject $H_0$** (462.9631 >> 0.918)
* **Significance Level 1.0%:** Critical Value = 1.091 --> **Reject $H_0$** (462.9631 >> 1.091)

In all significance levels, the Anderson-Darling test statistic is vastly larger than the critical values. This overwhelming evidence consistently leads to the **rejection of the null hypothesis** ($H_0$), which posits that the data follows a normal distribution. Thus, the Anderson-Darling test also strongly confirms that `Blindness_Duration` is **not normally distributed**.

## Conclusion

Based on the visual inspections of the histogram, boxplot, and Q-Q plot, along with the results from both the Kolmogorov-Smirnov and Anderson-Darling normality tests, it is unequivocally concluded that the `Blindness_Duration` variable **does not follow a normal distribution**. The data is characterized by a strong right-skewness, with a concentration of lower duration values and a long tail extending to higher durations. 