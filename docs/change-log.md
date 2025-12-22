# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.7.0] - 2025-07-08

### Added
- **Phase 3: Supervised Learning Module**
  - Complete supervised learning curriculum with 5 notebooks

- **Linear Regression** (`02-supervised-learning/01_linear_regression.ipynb`)
  - 31-cell comprehensive notebook
  - Simple and multiple linear regression
  - Gradient descent from scratch
  - Closed-form (normal equation) solution
  - Regularization (Ridge, Lasso, ElasticNet)
  - Regression assumptions and diagnostics
  - Polynomial regression

- **Logistic Regression** (`02-supervised-learning/02_logistic_regression.ipynb`)
  - 26-cell comprehensive notebook
  - Binary and multiclass classification
  - Sigmoid function and decision boundaries
  - Regularization and hyperparameter tuning
  - Evaluation metrics (precision, recall, F1, ROC-AUC)

- **Decision Trees & Random Forests** (`02-supervised-learning/03_decision_trees_random_forests.ipynb`)
  - 23-cell comprehensive notebook
  - Decision tree visualization
  - Gini impurity and entropy
  - Random Forest bagging
  - Feature importance analysis
  - Hyperparameter tuning

- **Support Vector Machines** (`02-supervised-learning/04_svm.ipynb`)
  - 24-cell comprehensive notebook
  - SVM theory (maximum margin, support vectors)
  - Kernel trick (linear, RBF, polynomial)
  - C and gamma parameter tuning
  - SVR for regression
  - Feature scaling importance

- **Gradient Boosting** (`02-supervised-learning/05_gradient_boosting.ipynb`)
  - 23-cell comprehensive notebook
  - Boosting vs bagging concepts
  - Gradient boosting from scratch
  - XGBoost and LightGBM
  - Hyperparameter tuning strategies
  - Model comparison

- **Supervised Learning Helpers** (`src/ml_core/supervised.py`)
  - `evaluate_classification()` - comprehensive classification metrics
  - `evaluate_regression()` - comprehensive regression metrics
  - `plot_learning_curve()` - bias/variance diagnosis
  - `plot_validation_curve()` - hyperparameter visualization
  - `plot_roc_curves()` - ROC curve comparison
  - `get_feature_importance_df()` - feature importance extraction
  - `compare_models()` - cross-validation model comparison
  - `create_baseline_models()` - baseline model creation

- **Supervised Learning Tests** (`tests/unit/test_supervised.py`)
  - 15 unit tests for supervised helpers

### Changed
- Updated `src/ml_core/__init__.py` with supervised module exports
- Total test count: 167 passing tests

## [1.6.0] - 2025-07-08

### Added
- **Scikit-Learn Introduction Module** (`01-fundamentals/05_sklearn_introduction.ipynb`)
  - 52-cell comprehensive scikit-learn notebook
  - Part 1: Introduction to Scikit-Learn (Estimator API)
  - Part 2: Loading and Exploring Datasets (built-in, synthetic)
  - Part 3: Train/Test Split (stratification, reproducibility)
  - Part 4: First Classification Model (KNN, comparing classifiers)
  - Part 5: Data Preprocessing (scaling, encoding, imputation)
  - Part 6: Pipelines (chaining, ColumnTransformer)
  - Part 7: Cross-Validation (K-Fold, StratifiedKFold, multi-metric)
  - Part 8: Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV)
  - Part 9: Practice Exercises & Summary

- **Scikit-Learn Helpers Module** (`src/utils/sklearn_helpers.py`)
  - `quick_train_test_split()` - train/test split with sensible defaults
  - `evaluate_classifier()` - comprehensive classifier evaluation
  - `evaluate_regressor()` - comprehensive regressor evaluation
  - `compare_classifiers()` - cross-validation comparison
  - `create_preprocessing_pipeline()` - mixed data preprocessing
  - `quick_grid_search()` - grid search with defaults
  - `learning_curve_data()` - generate learning curve data
  - `get_feature_importance()` - extract feature importance
  - `detect_data_leakage()` - check for train/test leakage
  - `model_summary()` - get fitted model summary

- **Scikit-Learn Tests** (`tests/unit/test_sklearn_helpers.py`)
  - 29 comprehensive unit tests for sklearn helpers
  - 100% function coverage

### Changed
- Updated `src/utils/__init__.py` with sklearn module exports
- Total test count: 152 passing tests

## [1.5.0] - 2025-07-08

### Added
- **Statistics for ML Module** (`01-fundamentals/04_statistics_for_ml.ipynb`)
  - 62-cell comprehensive statistics notebook
  - Part 1: Descriptive Statistics (mean, median, mode, variance, std)
  - Part 2: Shape Analysis (skewness, kurtosis, percentiles, quartiles)
  - Part 3: Probability Fundamentals (basic rules, conditional probability, Bayes' theorem)
  - Part 4: Probability Distributions (Normal, Binomial, Poisson, Uniform)
  - Part 5: Central Limit Theorem (demonstrations and practical implications)
  - Part 6: Hypothesis Testing (t-tests, p-values, effect size)
  - Part 7: Correlation & Covariance (Pearson, Spearman, significance testing)
  - Part 8: Practice Exercises & Summary

- **Stats Helpers Module** (`src/utils/stats_helpers.py`)
  - `describe_distribution()` - comprehensive statistics summary
  - `check_normality()` - Shapiro-Wilk normality test
  - `cohens_d()` - effect size calculation
  - `interpret_effect_size()` - effect size interpretation
  - `compare_two_groups()` - t-test with effect size
  - `calculate_confidence_interval()` - confidence interval estimation
  - `correlation_with_pvalue()` - correlation with significance
  - `find_outliers_zscore()` - z-score outlier detection
  - `find_outliers_iqr()` - IQR method outlier detection
  - `bootstrap_mean()` - bootstrap confidence intervals

- **Stats Tests** (`tests/unit/test_stats_helpers.py`)
  - 28 comprehensive unit tests for stats helpers
  - 100% function coverage

### Changed
- Updated `src/utils/__init__.py` with stats module exports

## [1.4.0] - 2025-07-08

### Added
- **Data Visualization Module** (`01-fundamentals/03_data_visualization.ipynb`)
  - 52-cell comprehensive visualization notebook
  - Matplotlib and Seaborn fundamentals
  - Distribution, relationship, categorical plots
  - Heatmaps and correlation matrices
  - ML-specific visualizations (confusion matrix, feature importance)
  - Customization and best practices
  - Practice exercises

- **Visualization Helpers Module** (`src/utils/visualization_helpers.py`)
  - 10 reusable plotting functions
  - EDA dashboard creation
  - Quick plotting utilities

- **Visualization Tests** (`tests/unit/test_visualization_helpers.py`)
  - 31 unit tests for visualization helpers

## [1.3.0] - 2025-07-07

### Added
- **Pandas Data Manipulation** (`01-fundamentals/02_pandas_data_manipulation.ipynb`)
  - 67-cell comprehensive Pandas notebook
  - DataFrames, Series, indexing, filtering
  - Data cleaning, groupby, merging
  - Time series and practical exercises

## [1.2.0] - 2025-07-07

### Added
- **NumPy Fundamentals** (`01-fundamentals/01_numpy_fundamentals.ipynb`)
  - 28KB comprehensive NumPy notebook
  - Arrays, operations, broadcasting, linear algebra

## [1.1.0] - 2025-07-06

### Added
- Project infrastructure
- Testing framework with pytest
- CI/CD with GitHub Actions
- Pre-commit hooks
- Documentation structure

## [1.0.0] - 2025-07-06

### Added
- Initial project setup
- Virtual environment configuration
- Base dependencies (NumPy, Pandas, Matplotlib, Seaborn)
- Project structure with notebooks, src, tests, docs folders
