# Comparison of Machine Learning Algorithms for Iyer, Cho, and YaleB Datasets

## Project Description

This project evaluates the performance of three machine learning algorithms:
1. **Logistic Regression (LR)**
2. **Random Forest (RF)**
3. **Convolutional Neural Network (CNN)**

The algorithms are applied to three datasets with varying characteristics:
- **Iyer**: Gene sequence dataset with 12 features and 11 classes.
- **Cho**: Gene sequence dataset with 16 features and 5 classes.
- **YaleB**: Grayscale human face dataset with 32x32 features and 38 classes.

### Objectives
- Compare the performance of the algorithms across datasets with distinct data types and feature structures.
- Identify the strengths and weaknesses of each algorithm in handling different data characteristics.

## Key Findings

| Algorithm   | Dataset   | Accuracy | F1 Score | AUC   |
|-------------|-----------|----------|----------|-------|
| Logistic Regression | Iyer | 0.799    | 0.794    | 0.882 |
| Random Forest       | Iyer | 0.791    | 0.790    | 0.873 |
| CNN                 | Iyer | 0.411    | 0.151    | 0.770 |

- **Logistic Regression**: Best performance for low-dimensional data (Iyer and Cho datasets).
- **Random Forest**: Balanced performance, excelling in datasets with moderate complexity (Cho dataset).
- **CNN**: Superior for high-dimensional data, particularly image data (YaleB dataset).

### Pros and Cons of Algorithms
- **Logistic Regression**:
  - Pros: Simple, fast, interpretable, works well for low-dimensional data.
  - Cons: Poor handling of high-dimensional and nonlinear data.
- **Random Forest**:
  - Pros: Handles nonlinearity and missing data, provides feature importance.
  - Cons: Computationally intensive, prone to overfitting with many trees.
- **CNN**:
  - Pros: Excels in high-dimensional and image data, automatically learns features.
  - Cons: Requires significant computational resources, challenging to tune.

## Code Overview

The implementation is provided in `code.Rmd`, written in **R** using various libraries:
- **Preprocessing**: `dplyr`, `caret`
- **Modeling**:
  - Logistic Regression: `nnet`
  - Random Forest: `randomForest`
  - CNN: `keras`, `tensorflow`
- **Evaluation**: `pROC`, `MLmetrics`

### Structure
1. **Data Loading**: Import datasets and perform initial exploratory data analysis.
2. **Preprocessing**:
   - Cross-validation for Iyer and Cho datasets.
   - PCA for dimensionality reduction in YaleB dataset.
3. **Model Training**:
   - Logistic Regression: `nnet()`
   - Random Forest: `randomForest()`
   - CNN: `keras_model_sequential()`
4. **Evaluation**:
   - Metrics: Accuracy, F1 Score, AUC.
   - Hyperparameter tuning with grid search.

## How to Run

1. Install required R libraries:
install.packages(c("dplyr", "caret", "pROC", "MLmetrics", "randomForest"))

For CNN:
install.packages("keras") library(keras) install_keras()

2. Open the `code.Rmd` file in RStudio.
3. Run the code sequentially to preprocess data, train models, and evaluate results.

## Learning Outcomes

This project highlights:
- The importance of algorithm selection based on data characteristics.
- The challenges of training and tuning CNNs for non-image data.
- The utility of logistic regression and random forest for gene sequence data.

## References

- **Datasets**: Iyer and Cho datasets from the UCI Machine Learning Repository; YaleB dataset for grayscale face images.
- **Libraries**: `keras`, `tensorflow`, `randomForest`, `caret`, `MLmetrics`.

For more details, refer to the project report.


