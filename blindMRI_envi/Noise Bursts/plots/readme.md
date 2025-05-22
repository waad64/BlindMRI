# Analysis of NBM Values Dataset

This document provides an in-depth analysis of the "Noise Bursts per Minute (NBM)" dataset, including descriptive statistics, distribution visualization, and statistical tests for distribution type and outlier detection.

## 1\. Dataset Overview

The dataset comprises `6776` individual measurements of Noise Bursts per Minute (NBM).

### Basic Statistics:

| Statistic            | Value      |
| :------------------- | :--------- |
| **Count** | `6776.00`  |
| **Mean** | `4.59`     |
| **Standard Deviation** | `3.65`     |
| **Minimum** | `0.00`     |
| **25th Percentile (Q1)** | `1.25`     |
| **50th Percentile (Median)** | `3.99`     |
| **75th Percentile (Q3)** | `7.82`     |
| **Maximum** | `12.00`    |

The NBM values range from 0 to 12. The mean NBM is approximately 4.59, with a standard deviation of 3.65. The median (3.99) is slightly lower than the mean, which can sometimes indicate a right skew in the distribution.

## 2\. Distribution Analysis

### 2.1. Histogram and Kernel Density Estimate (KDE)

*Figure 1* displays the histogram and Kernel Density Estimate (KDE) of the NBM values. The distribution is distinctly **multi-modal**, exhibiting several prominent peaks. Specifically, there's a large peak around 0-2 NBM, another significant peak around 4-5 NBM, and a third, broader peak in the 8-12 NBM range. This multi-modality strongly suggests that the NBM data is composed of distinct subgroups or phenomena, rather than following a single, continuous distribution.

### 2.2. Quantile-Quantile (Q-Q) Plot vs Normal Distribution

*Figure 3* presents a Q-Q plot comparing the observed NBM values against a theoretical normal distribution. The blue data points deviate significantly from the red straight line, forming an "S"-shape. This pronounced departure, especially at both the lower and upper tails, provides clear visual evidence that the NBM values **do not follow a normal distribution**.

## 3\. Statistical Tests for Distribution

### 3.1. Kolmogorov-Smirnov (K-S) Test Against Uniform(0,12)

  * **Statistic:** `0.2337`
  * **P-value:** `0.0000`

The Kolmogorov-Smirnov test was conducted to determine if the NBM values are uniformly distributed between 0 and 12. The resulting p-value of `0.0000` is extremely small, leading to a **strong rejection of the null hypothesis** that the data is uniformly distributed. This finding is consistent with the highly non-uniform appearance of the histogram.

### 3.2. Anderson-Darling Test for Normality

  * **Test Statistic:** `208.9541`

Critical values for various significance levels:

  * `15.0%` critical value: `0.5760` -\> **Reject normality**
  * `10.0%` critical value: `0.6560` -\> **Reject normality**
  * `5.0%` critical value: `0.7870` -\> **Reject normality**
  * `2.5%` critical value: `0.9170` -\> **Reject normality**
  * `1.0%` critical value: `1.0910` -\> **Reject normality**

The Anderson-Darling test statistic (`208.9541`) is orders of magnitude larger than all critical values at typical significance levels. This overwhelmingly indicates that we **reject the null hypothesis of normality** for the NBM values. This statistical confirmation aligns with the visual evidence from the Q-Q plot and the multi-modal histogram.

## 4\. Outlier Detection

### 4.1. Boxplots and IQR Method

*Figure 2* and *Figure 4* display boxplots of the NBM values. The Interquartile Range (IQR) method was applied to identify any potential outliers.

  * **Detected Outliers:** `0`
  * **Outlier Values (first 10, if any):** `Series([], Name: NBM, dtype: float64)`

Based on the IQR method, **no outliers were detected** in the NBM dataset. This suggests that while the distribution is complex and non-normal, all data points fall within the expected range for the given spread, according to the IQR criterion. The boxplot whiskers extend to the minimum (0) and maximum (12) values, which is consistent with the absence of IQR-defined outliers.