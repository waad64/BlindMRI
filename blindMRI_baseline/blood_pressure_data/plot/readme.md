# 📋 Blood Pressure Data Analysis Report

This document presents a comprehensive statistical and visual assessment of the **Systolic** and **Diastolic Blood Pressure** distributions in the dataset. The focus is on understanding the shape of the data, central tendencies, spread, and normality—crucial for guiding subsequent analysis and modeling decisions.

---

## 1. Overview of Data

- **Data Source:** Curated blood pressure measurements from the enhanced health dataset.
- **Variables:**  
  - Systolic Blood Pressure (SBP)  
  - Diastolic Blood Pressure (DBP)  
- **Sample Size:** 888 patients (after cleaning and sampling)

---

## 2. Systolic Blood Pressure (SBP)

### Distribution Characteristics

- The **histogram** reveals a roughly bell-shaped, unimodal distribution.
- The **peak (mode)** lies near 130-135 mmHg, indicating the most common SBP range.
- Values range approximately from **105 mmHg to 150 mmHg**, showing reasonable physiological variation.
- The distribution shows a slight **negative (left) skew**, evidenced by a longer tail on the lower side.
- The **Kernel Density Estimate (KDE)** overlays confirm the smooth bell shape with minor left skew.

### Boxplot Summary

- The **interquartile range (IQR)** spans from about 115 to 140 mmHg.
- The **median** is slightly above the center of the box, consistent with mild left skew.
- Whiskers indicate no extreme outliers outside 1.5x IQR.

### Q-Q Plot Interpretation

- Data points deviate from the theoretical normal line, especially in the tails.
- This indicates **heavier tails** than expected under a normal distribution.

### Normality Test Results

| Test                   | Statistic / p-value | Interpretation                         |
|------------------------|---------------------|--------------------------------------|
| Shapiro-Wilk           | p = 0.0002          | Strong rejection of normality        |
| Anderson-Darling       | 0.9562              | Statistic exceeds critical values → reject normality |
| Kolmogorov-Smirnov     | p = 0.1890          | Fail to reject normality (contradictory result) |

### Summary

Despite the Kolmogorov-Smirnov test suggesting normality, both the **Shapiro-Wilk** and **Anderson-Darling** tests and visual diagnostics strongly indicate that the SBP data **do not perfectly follow a normal distribution**. The slight negative skew and heavy tails should be accounted for in further analyses.

---

## 3. Diastolic Blood Pressure (DBP)

### Distribution Characteristics

- The **histogram** shows a unimodal but **right-skewed** distribution.
- The most frequent diastolic BP values cluster around **80-85 mmHg**.
- Values range roughly from **65 mmHg to 95 mmHg**.
- The KDE curve indicates the right tail extends toward higher values.
- The distribution is less symmetrical compared to SBP.

### Boxplot Summary

- IQR covers approximately 75 to 85 mmHg.
- The median is shifted towards the right within the box, supporting positive skewness.
- Several **outliers** appear on the lower end, consistent with a tail towards lower DBP values.

### Q-Q Plot Interpretation

- Significant deviation from the normality line, especially at both extremes.
- The plot’s S-shape confirms skewness and heavier tails.

### Normality Test Results

| Test                   | Statistic / p-value | Interpretation                    |
|------------------------|---------------------|---------------------------------|
| Shapiro-Wilk           | p ≈ 0.0000          | Strong rejection of normality   |
| Anderson-Darling       | 4.7930              | Well above critical values → reject normality |
| Kolmogorov-Smirnov     | p = 0.0003          | Reject normality                |

### Summary

All tests and visualizations consistently confirm that the DBP distribution **significantly deviates from normality** with pronounced right skew and outliers. 

---

## 4. Overall Conclusions

-Realistic ranges: Both Systolic (105–150 mmHg) and Diastolic (65–95 mmHg) blood pressures are within expected physiological limits — no wild outliers throwing off sanity.

-Sample size: 888 is a solid number, giving - decent statistical power and variability.

-Clear distribution patterns: we  have well-defined peaks and spreads, meaning the data isn’t just noise — it has structure.

-Balanced in range:  SBP and DBP cover typical adult BP ranges without suspicious gaps

-GANs Don’t Need Normality: They learn the true data distribution (including skew/tails). The non-normal data  actually improve GAN realism.
---

