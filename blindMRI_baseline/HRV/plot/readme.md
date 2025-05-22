# HRV (RMSSD) Data Quality and Normality Report

---

##  Visual Exploration

- **Histogram + KDE Plot:**  
  The RMSSD distribution clearly deviates from normality, showing possible skewness or multiple modes rather than a classic bell curve. The KDE highlights underlying density variations.

- **Boxplot:**  
  No extreme outliers detected beyond the usual range, confirming the z-score analysis (0 outliers with |z| > 3). The spread reflects natural physiological variability.

- **Q-Q Plot:**  
  Strong deviations from the theoretical normal line, especially at the tails, visually confirming that the RMSSD data is not normally distributed.

---

##  Outlier Analysis

- **Z-score method:**  
  Zero outliers beyond 3 standard deviations! data are clean, without extreme values that could bias analysis.

---

## 📈 Normality Tests Summary

| Test               | Statistic  | Critical Value(s) / p-value | Interpretation                               |
|--------------------|------------|-----------------------------|----------------------------------------------|
| Anderson-Darling    | 110.1613   | Critical values ~0.5 - 1.1   | Statistic >> critical values ⇒ **reject normality** |
| Kolmogorov-Smirnov  | 0.32250    | p-value = 0.00000            | p-value << 0.05 ⇒ **reject normality**       |

- Both tests strongly reject the null hypothesis of normality.  
- The Anderson-Darling statistic is **way above** critical thresholds, signaling a strong deviation from normal distribution.

---

##  Conclusion

- HRV data are **clean and reliable**, but clearly **not normally distributed**.  
- No major outliers, which is excellent for robust downstream analysis.  
- The skewed distribution calls for careful choice of analysis methods.

---


> In short: the HRV data are **healthy and ready to go**.  


---

