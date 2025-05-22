# Metadata Statistical Assessment Report 
Demographic metadata (Age & Gender) extracted from WESAD-derived sequences.

---

## 1. Objective

Evaluate the quality and representativeness of the demographic metadata—specifically patient age and gender. The data were curated from multiple sequences, filtered for valid entries (ages 6 to 70), and balanced across age groups.

---

## 2. Data Preprocessing Summary

- Source: Extracted from WESAD-like sequences, shuffled before sampling
- Selected columns: `Patient Age`, `Patient Gender`
- Filters applied:
  - Removed missing values
  - Age constrained between 6 and 70 years
  - Gender encoded: Female = 1, Male = 0
- Stratified sampling to balance age groups
- Final sample size: **888 patients**

---

## 3. Descriptive Statistics

| Feature        | Count | Mean | Std  | Min | 25% | Median | 75% | Max |
|----------------|-------|------|------|-----|-----|--------|-----|-----|
| Patient Age    | 888   | ~46  | ~15  | 6   | 25  | 46     | 59  | 70  |
| Patient Gender | 888   | -    | -    | M: 520 | - | - | - | F: 368 |

- Gender ratio: ~58.5% Male, 41.5% Female
- Age range: 6 to 70 years, with bimodal distribution peaks

---

## 4. Normality Tests on Age

### Anderson-Darling Test

- Statistic: 20.6728  
- Critical values at significance levels (15%, 10%, 5%, 2.5%, 1%): 0.5730, 0.6530, 0.7830, 0.9140, 1.0870  
- **Interpretation:** Statistic far exceeds all critical values → **Reject null hypothesis** of normality.

### Kolmogorov-Smirnov Test

- Statistic: 0.1252  
- p-value: < 0.0001  
- **Interpretation:** p-value < 0.05 → **Reject null hypothesis** of normality.

---

## 5. Visual Diagnostics

- **Histogram + KDE:** Shows bimodal peaks (~15–20 and 65–70 years)
- **KDE Plot:** Smooth density confirming bimodality
- **Boxplot:** Median age ~46, interquartile range 25–59
- **Q-Q Plot:** Deviations from normality especially at distribution tails
- **KDE by Gender:** Similar distributions with slight difference in spread

---

## 6. Demographic Distribution Insights

### Gender Distribution

- Slight imbalance with males (58.5%) more represented than females (41.5%)

### Age Distribution

- Bimodal with peaks in younger (~15-20) and older (~65-70) age ranges
- Mid-range ages (~20-55) fairly uniformly distributed
- No strong skew or outliers detected

### Age vs Gender

- Median age similar for both genders (~46 years)
- Slightly tighter age spread in females than males
- Overall similar age range (6–70 years)

---

## 7. Interpretation & Recommendations

| Aspect             | Verdict            | 
|--------------------|--------------------|
| Data Cleanliness   |  Good              | 
| Gender Balance     | Moderate Imbalance | 
| Age Distribution   | Representative     |     
| Normality          | Not Normal         | 
| Bias Check         | Acceptable         | 

---

## 8. Final Remarks

The dataset is well-prepared and demographically diverse. Although the age data is not normally distributed, this is common in real-world datasets and does not hinder most machine learning applications. 

**Conclusion:** The metadata is robust and suitable for downstream analysis and modeling.

---

