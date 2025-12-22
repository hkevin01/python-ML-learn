# Change Log

## v1.4.0 - Data Visualization Module (Current)
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
