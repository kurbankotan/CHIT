# Cyclical Hybrid Imputation Technique (CHIT)

Implementation code for the paper published in *Scientific Reports*:

> **Kotan, K., Kırışoğlu, S.** Cyclical hybrid imputation technique for missing values in data sets. *Sci Rep* **15**, 6543 (2025).  
> DOI: [10.1038/s41598-025-90964-7](https://doi.org/10.1038/s41598-025-90964-7)

---

## What is CHIT?

CHIT (Cyclical Hybrid Imputation Technique) is a novel missing data imputation algorithm that combines **column-based** (temporary fill) and **row-based** (permanent fill) imputation methods cyclically. Unlike MICE or KNNImputer, CHIT dynamically changes the imputation method for each row and column operation, estimating missing values with smaller error margins.

---

## Algorithm

### Step-by-step

**Input:** Dataset with missing values, a column-based method, a row-based regression model

1. **Partition** the dataset:
   - **Y** → complete rows (no missing values)
   - **X** → rows containing at least one missing value

2. **Select** the row in X with the fewest missing values (**i\***):

$$i^* = \arg\max_i \sum_{j=1}^{p} M_{ij}$$

3. **Temporarily fill** the missing cells of i\* using the selected column-based method (mean, median, mode, ffill, bfill, KNN, or interpolation).

4. **Permanently fill** each missing cell, starting from the column with the most missing values across the dataset (**j\***):

$$j^* = \arg\max_j \sum_{i=1}^{n}(1 - M_{ij}), \quad \forall j \in J_{\text{NaN}}$$

   - Train the row-based regression model on **Y**
   - Predict and permanently replace the temporary value in cell (i\*, j\*)
   - Move to the next j\* until the row is complete

5. **Move** the completed row from X to Y.

6. **Repeat** from Step 2 until X is empty.

---

## Imputation Methods

### Column-based methods (temporary fill — Step 3)

| `col_strategy` | Method |
|---|---|
| `'mean'` | Column mean of complete rows (Y) |
| `'median'` | Column median of complete rows (Y) |
| `'mode'` | Most frequent value in the column |
| `'ffill'` | Last known value in the column |
| `'bfill'` | First known value in the column |
| `'knn'` | KNNImputer (n_neighbors=5) applied to Y + current row |
| `'interpolation'` | Linear interpolation over Y + current row |

### Row-based models (permanent fill — Step 4)

Configurable via `ROW_MODEL`. Candidates from the paper:

| Model | sklearn class |
|---|---|
| K-Nearest Neighbors | `KNeighborsRegressor` |
| Linear Regression | `LinearRegression` |
| Support Vector Regression | `SVR` |
| Decision Tree | `DecisionTreeRegressor` |
| Random Forest | `RandomForestRegressor` |
| Hist Gradient Boosting | `HistGradientBoostingRegressor` |
| Deep Learning | Keras `Sequential` |

---

## Experiments

The code runs all **7 column-based methods** automatically for a chosen row-based model, producing a full comparison table — matching the experimental structure of the paper.

```
7 col_strategies × 8 classifiers = 56 experiments per row model
```

### Classifier models (evaluation phase)

Each classifier is tuned with `GridSearchCV (cv=2)` and evaluated with 5-fold cross-validation:

| Classifier | Hyperparameters searched |
|---|---|
| KNN | n_neighbors, weights, leaf_size |
| Logistic Regression | C, penalty, random_state |
| SVC | C, gamma |
| Decision Tree | criterion, max_depth |
| Random Forest | bootstrap, max_depth, max_features, min_samples_leaf, n_estimators |
| Gaussian Naïve Bayes | var_smoothing |
| MLP | solver, alpha, activation |
| Deep Learning (Keras) | Dense(1000→500→300→2), Dropout(0.2), Adam, 10 epochs |

---

## Dataset

**Chronic Kidney Disease (CKD)** — UCI Machine Learning Repository  
DOI: [10.24432/C5G020](https://doi.org/10.24432/C5G020)

- 400 patient records, 24 clinical features + 1 label column
- Label: `ckd` (1) / `notckd` (0)
- High rate of missing values across numerical and categorical features

### Preprocessing

| Feature type | Method |
|---|---|
| Categorical (rbc, pc, pcc, ...) | `SimpleImputer(most_frequent)` + `LabelEncoder` |
| Numerical (age, bp, sg, ...) | **CHIT** (column temp fill + row-based regression) |
| Target (classification) | Included in categorical imputation |

---

## How to Run

### Google Colab

1. Upload `kidney_disease.csv` to Google Drive:
   ```
   MyDrive/Colab Notebooks/datasets/kidney_disease/kidney_disease.csv
   ```

2. Open `Chronic_Kidney_Disease(Kronik_Böbrek_Hastalığı).ipynb` in [Google Colab](https://colab.research.google.com).

3. To change the **row-based model**, edit line:
   ```python
   ROW_MODEL = KNeighborsRegressor(n_neighbors=5)
   ```

4. To show **confusion matrices**, change:
   ```python
   run_classifiers(..., show_plots=True)
   ```

5. Run all cells. The final output is a ranked comparison table across all 7 column-based methods.

> All required libraries (scikit-learn, TensorFlow, pandas, seaborn) are pre-installed in Colab.

---

## Output

Two summary tables are displayed at the end:

**Full results table** — one row per (col\_strategy × classifier) combination:

| Col\_Strategy | Classifier | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| knn | SVC | 1.00 | 1.00 | 1.00 | 1.00 |
| mean | LR | 0.99 | 0.99 | 0.99 | 0.99 |
| ... | ... | ... | ... | ... | ... |

**Best per strategy** — peak F1 and Accuracy for each column-based method.

---

## Files

| File | Description |
|---|---|
| `Chronic_Kidney_Disease(Kronik_Böbrek_Hastalığı).ipynb` | Main Jupyter notebook (Google Colab) |
| `chronic_kidney_disease(kronik_böbrek_hastalığı).py` | Equivalent Python script |

---

## Citation

```bibtex
@article{kotan2025chit,
  title   = {Cyclical hybrid imputation technique for missing values in data sets},
  author  = {Kotan, Kurban and K{\i}r{\i}{\c{s}}o{\u{g}}lu, Serdar},
  journal = {Scientific Reports},
  volume  = {15},
  pages   = {6543},
  year    = {2025},
  doi     = {10.1038/s41598-025-90964-7}
}
```

---

## Authors

- **Kurban Kotan** — Duzce University, Graduate School of Education
- **Serdar Kırışoğlu** — Duzce University, Department of Computer Engineering
