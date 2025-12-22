# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
