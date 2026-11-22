# Breast Cancer Classification with Scikit-learn

A small machine learning project that classifies breast tumours as malignant or benign from measurements of cell nuclei in digitised biopsy images.

**Author:** Cassandra Purity Ronnie (biomedical scientist, molecular biology and bioinformatics)

## Question

Can a simple machine learning model distinguish malignant from benign tumours using numeric features extracted from biopsy images, and how well does it do on data it has not seen?

## Data

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), loaded with `sklearn.datasets.load_breast_cancer`. Originally from the UCI Machine Learning Repository.
- **Size:** 569 samples, 30 numeric features (mean, standard error and "worst" values of 10 nuclear measurements such as radius, texture and concavity). No missing values.
- **Classes:** 212 malignant (37.3%) and 357 benign (62.7%).
- **Label coding:** scikit-learn codes malignant as 0. I flipped this so malignant = 1, making it the positive class, because missing a malignant case is the more serious error.

## Method

1. Stratified 80/20 train/test split (455 training and 114 test samples, `random_state=42`).
2. Compared two models with 5-fold stratified cross-validation on the **training data only**:
   - Logistic regression with feature scaling (in a pipeline to avoid data leakage)
   - Random forest (300 trees)
3. Chose the model with the better cross-validated recall and F1 for the malignant class.
4. Fitted it on all training data and evaluated it **once** on the held-out test set.

## Results

**Cross-validation (training set, 5 folds):**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | 0.974 | 0.977 | 0.953 | 0.964 |
| Random forest | 0.960 | 0.958 | 0.935 | 0.946 |

**Held-out test set (logistic regression, malignant = positive class):**

| Metric | Score |
|---|---|
| Accuracy | 0.965 |
| Precision | 0.975 |
| Recall | 0.929 |
| F1 | 0.951 |

On the 114 test samples the model made 4 errors: 3 malignant cases classified as benign (false negatives) and 1 benign case classified as malignant (false positive).

![Confusion matrix](figures/confusion_matrix.png)

The features with the largest coefficients included worst texture, radius error, worst symmetry and mean concave points (see `figures/top_features.png`). This shows what the model relies on in this dataset and does not show biological causation. Many features are strongly correlated, so individual coefficients should be read with caution.

![Top features](figures/top_features.png)

## Limitations

- **Small, single dataset:** 569 samples from one source, with no external validation, so results may not generalise to other hospitals or imaging setups.
- **Small test set:** With 114 test samples each error moves the metrics noticeably, so the scores are estimates with uncertainty.
- **Not a clinical tool:** This is a learning project using pre-extracted features. It has not been validated for any medical use.
- **Limited tuning:** Only two models and mostly default settings were compared.
- **Features are pre-engineered:** The project does not cover image processing.

## Possible next steps

Hyperparameter tuning, more models, probability calibration and threshold choice (to trade precision against recall), and testing on a separate breast cancer dataset.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook breast_cancer_classifier.ipynb
```

## Files

- `breast_cancer_classifier.ipynb`: the full analysis with outputs
- `figures/`: saved plots
- `requirements.txt`: Python dependencies
