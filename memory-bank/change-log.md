# Change Log

## v2.0.0 - Project Complete: MLOps & Production (Current)
**Date**: December 22, 2025

### 🎉 PROJECT MILESTONE: COMPLETE!
This release marks the completion of the entire 9-phase machine learning study guide curriculum.

### Added - Phase 9: MLOps & Production
- `08-mlops/01_model_serving_fastapi.ipynb` - REST API model serving
  - FastAPI application structure
  - Pydantic request/response validation
  - ModelManager class for loading/inference
  - Health checks, batch predictions, benchmarking
- `08-mlops/02_docker_containerization.ipynb` - Docker for ML
  - Basic, multi-stage, and GPU Dockerfiles
  - docker-compose.yml configuration
  - Optimization techniques
- `08-mlops/03_experiment_tracking.ipynb` - Experiment management
  - ExperimentTracker class (params, metrics, artifacts)
  - ModelRegistry with versioning and stage transitions
  - Experiment comparison utilities
- `08-mlops/04_cicd_pipelines.ipynb` - Automated pipelines
  - GitHub Actions workflows (test, train, deploy)
  - Model validation scripts
  - Pre-commit configuration
- `08-mlops/05_model_monitoring.ipynb` - Production monitoring
  - PredictionLogger for tracking predictions
  - DriftDetector with KS test for data drift
  - PerformanceMonitor for metrics tracking
  - AlertManager with severity levels
  - ModelMonitor comprehensive system

### Project Statistics
- **Total Notebooks**: 38
- **Total Tests**: 362 passing
- **Phases Completed**: 9/9 (100%)
- **Estimated Study Hours**: 560+

---

## v1.9.0 - End-to-End Projects Complete
**Date**: December 22, 2025

### Added - Phase 8: End-to-End Projects
- `07-projects/01_house_price_prediction.ipynb` - Regression project
- `07-projects/02_customer_churn_prediction.ipynb` - Classification with imbalanced data
- `07-projects/03_image_classification_app.ipynb` - Computer vision project
- `07-projects/04_sentiment_analysis_pipeline.ipynb` - NLP classification
- `07-projects/05_recommendation_system.ipynb` - Collaborative/content filtering

---

## v1.4.0 - Data Visualization Module
**Date**: $(date +%Y-%m-%d)

### Added
- `01-fundamentals/03_data_visualization.ipynb` - Complete visualization notebook (52 cells)
  - Matplotlib fundamentals (figure anatomy, subplots)
  - Distribution plots (histogram, KDE, box plots, violin plots)
  - Relationship plots (scatter, pair plots, joint plots, hexbin)
  - Categorical plots (bar, count, strip, swarm)
  - Heatmaps and correlation analysis
  - ML-specific visualizations (confusion matrix, learning curves, feature importance)
  - Customization and best practices
  - Practice exercises
- `src/utils/visualization_helpers.py` - Reusable plotting functions
  - `plot_distribution()` - Distribution plots with stats
  - `plot_correlation_heatmap()` - Correlation heatmaps with threshold detection
  - `plot_missing_values()` - Missing value visualization
  - `plot_class_balance()` - Class imbalance visualization
  - `plot_feature_importance()` - Feature importance bar charts
  - `plot_confusion_matrix()` - Confusion matrix heatmaps
  - `create_eda_dashboard()` - Comprehensive EDA dashboard
  - Quick functions: `quick_hist()`, `quick_corr()`, `quick_missing()`
- `tests/unit/test_visualization_helpers.py` - 31 tests for visualization module

### Dependencies
- matplotlib 3.10.8
- seaborn 0.13.2

### Test Results
- **90 tests passing** (24 numpy + 21 pandas + 14 timer + 31 visualization)

---

## v1.3.0 - Pandas Module Complete
**Date**: 2025-01-XX

### Added
- `01-fundamentals/02_pandas_data_manipulation.ipynb` - Complete pandas notebook (67 cells)
- `src/utils/pandas_helpers.py` - 8 utility functions for data manipulation
- `tests/unit/test_pandas_helpers.py` - 21 tests for pandas helpers
- `.github/instructions/memory.instruction.md` - Copilot memory for work preferences

---

## v1.2.0 - NumPy Fundamentals
**Date**: 2025-01-XX

### Added
- `01-fundamentals/01_numpy_fundamentals.ipynb` - Complete NumPy notebook

---

## v1.1.0 - Project Infrastructure
**Date**: 2025-01-XX

### Added
- Project structure with src/, tests/, docs/ folders
- Timer utility module with tests
- pytest configuration
- Virtual environment setup

---

## v1.0.0 - Initial Setup
**Date**: 2025-01-XX

### Added
- README.md with project overview
- project-plan.md with detailed curriculum
- Basic folder structure
