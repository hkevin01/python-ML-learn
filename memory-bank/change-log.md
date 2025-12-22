# Change Log

All notable changes to the Python Machine Learning Study Guide project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Pandas Data Manipulation Notebook
- Data Visualization Notebook
- Statistics & Probability Notebook
- Feature Engineering Pipeline
- Additional deep learning notebooks
- Interactive web interface
- Video tutorials

---



## [1.3.0] - 2025-12-22

### Added - Pandas Data Manipulation Module

#### Date: December 22, 2025
#### Modified Components: Phase 2 Progress - Pandas Complete
#### Contributors: Project Maintainer

**Changes Made:**

**Pandas Fundamentals Notebook (67 cells):**
- Created comprehensive notebook: `01-fundamentals/02_pandas_data_manipulation.ipynb`
- Built incrementally in 6 parts to avoid length limits:
  - Part 1: Introduction + Series/DataFrame basics (16 cells)
  - Part 2: Data Loading + Selection/Indexing (11 cells)  
  - Part 3: Data Cleaning - Missing values, types, duplicates (12 cells)
  - Part 4: GroupBy Operations (8 cells)
  - Part 5: Merging and Joining DataFrames (7 cells)
  - Part 6: Practice Exercises + Summary (13 cells)

**Pandas Helper Module:**
- Created `src/utils/pandas_helpers.py` with 8 utility functions:
  - `load_csv_optimized()` - Optimized CSV loading for ML
  - `get_missing_info()` - Detailed missing value analysis
  - `fill_missing_by_group()` - Group-wise imputation
  - `create_dummy_variables()` - One-hot encoding
  - `detect_outliers_iqr()` - IQR-based outlier detection
  - `split_train_test()` - Train/test splitting with stratification
  - `reduce_memory_usage()` - Memory optimization via downcasting
  - `dataframe_info()` - Comprehensive DataFrame summary

**Unit Tests:**
- Created `tests/unit/test_pandas_helpers.py` (21 tests)
- All 59 tests passing (24 numpy + 21 pandas + 14 timer)

**Documentation:**
- Updated `src/utils/__init__.py` with pandas_helpers exports
- Created Copilot memory instructions for working in parts

**Dependencies Added:**
- pandas 2.3.3
- openpyxl 3.1.5 (Excel support)
- xlrd 2.0.2 (Legacy Excel support)

**Impact:**
- Phase 2 now 40% complete (2/5 notebooks)
- Solid pandas foundation for feature engineering
- Reusable helper functions for all future notebooks


---

## [1.2.0] - 2025-12-22

### Added - Comprehensive Documentation Update

#### Date: December 22, 2025
#### Modified Components: README.md, project-plan.md
#### Contributors: Project Maintainer

**Changes Made:**

**README.md Complete Rewrite (753 lines):**
- Added comprehensive project overview with badges
- Created "Why This Project?" section with problem/solution table
- Added detailed project goals with learning outcomes
- Included Mermaid architecture diagrams (all with dark backgrounds):
  - System Architecture flowchart
  - Learning Path mindmap
  - Core Skills mindmap
  - Gantt chart for project timeline
  - Pie chart for phase distribution
- Created complete Technology Stack section with tables:
  - Core Stack (Python, NumPy, Pandas, scikit-learn, PyTorch, TensorFlow)
  - Development Tools (pytest, Docker, VS Code)
  - Each technology includes version, purpose, and why chosen
- Added detailed Getting Started instructions (local and Docker)
- Created expandable Learning Path sections with:
  - Phase definitions and motivations
  - Step-by-step mechanisms
  - Mathematical formulations (KaTeX)
  - Implementation details
  - Measured impact metrics
- Added project structure visualization
- Included current progress table with visual indicators
- Added contributing guidelines and license info

**Project Plan Update (722 lines):**
- Added Executive Summary table
- Created ML Learning Gap problem visualization
- Added concept deep-dives for all phases:
  - Definition, motivation, mechanism
  - Mathematical formulations with LaTeX
  - Implementation details with code samples
  - Measured impact metrics
- Added detailed progress trackers per phase
- Included algorithm overviews with formulas:
  - Linear/Logistic Regression
  - Decision Trees & Random Forests
  - Neural Networks
- Added Success Metrics & KPIs tables
- Created Risk Management section
- Added Reference Materials with links

**Documentation Quality:**
- All Mermaid diagrams use dark backgrounds for readability
- Tables consistently formatted across both files
- Mathematical formulas in KaTeX format
- Comprehensive progress tracking with emoji indicators
- ASCII art for supplementary visualizations

**Impact:**
- Professional-grade project documentation
- GitHub-ready with proper Mermaid rendering
- Clear learning path visible at a glance
- Detailed technical specifications for every concept

---

## [1.1.0] - 2025-12-22

### Added - NumPy Fundamentals & Testing Framework

#### Date: December 22, 2025
#### Modified Components: Phase 1 Complete, Phase 2 Started
#### Contributors: Project Maintainer

**Changes Made:**

**Phase 1 Completion:**
- Created learning path folder structure (01-fundamentals through 07-projects)
- Set up pytest testing framework with conftest.py and pytest.ini
- Created virtual environment with core dependencies
- Validated Docker environment configuration

**Phase 2 Progress - NumPy Fundamentals:**
- Created comprehensive NumPy fundamentals notebook (`01-fundamentals/01_numpy_fundamentals.ipynb`)
  - Array creation and initialization
  - Indexing and slicing
  - Boolean indexing (essential for ML)
  - Array operations and ufuncs
  - Broadcasting
  - Linear algebra operations
  - Random number generation
  - Performance comparison (NumPy vs Python lists)
  - 5 practice exercises with solutions

- Created numpy_helpers utility module (`src/utils/numpy_helpers.py`)
  - normalize() - min-max, z-score, L1, L2 normalization
  - standardize() - z-score standardization shorthand
  - check_nan(), check_inf() - array validation
  - array_info() - comprehensive array statistics
  - safe_divide() - division with zero handling
  - clip_outliers() - percentile-based outlier clipping
  - moving_average() - sliding window average
  - train_test_split_indices() - data splitting

- Created unit tests (`tests/unit/test_numpy_helpers.py`)
  - 24 comprehensive test cases
  - 100% coverage of numpy_helpers functions

**Testing Notes:**
- All 38 unit tests passing
- pytest + pytest-cov configured
- Tests validate Timer utilities (14 tests)
- Tests validate NumPy helpers (24 tests)

**Technical Details:**
- Python 3.12.3 environment
- NumPy 2.4.0 installed
- pytest 9.0.2 with coverage plugin

**Impact:**
- Phase 1 infrastructure 100% complete
- Phase 2 NumPy fundamentals complete
- Solid foundation for remaining ML fundamentals

---

## [1.0.0] - 2025-12-22

### Added - Initial Project Setup

#### Date: December 22, 2025
#### Modified Components: Project Structure, Core Infrastructure
#### Contributors: Project Maintainer

**Changes Made:**
- Created comprehensive project structure with src layout
- Established memory-bank system for documentation
- Set up modular folder organization:
  - `src/` for source code (utils, models, data_processing, visualization)
  - `tests/` for unit and integration tests
  - `docs/` for documentation
  - `scripts/` for utility scripts
  - `data/` for datasets (raw, processed, external)
  - `assets/` for images and saved models
  - `docker/` for containerization
  - `configs/` for configuration files
  - `.github/` for GitHub workflows and templates
  - `.copilot/` for AI assistant configurations
  - `.vscode/` for VS Code settings

**Documentation Created:**
- `memory-bank/app-description.md` - Comprehensive project overview
- `README.md` - Project introduction and getting started guide
- Initial folder structure for organized learning path

**Testing Notes:**
- Project structure validated
- All directories created successfully
- Ready for content population

**Technical Details:**
- Python 3.8+ requirement
- Jupyter notebook-based learning approach
- Docker support for reproducible environments
- Modern development tooling configured

**Impact:**
- Provides clear learning path for ML beginners
- Modular structure allows easy expansion
- Production-ready code organization

---

## Template for Future Entries

```markdown
## [Version] - YYYY-MM-DD

### Added/Changed/Fixed/Removed

#### Date: Month Day, Year
#### Modified Components: [Component names]
#### Contributors: [Names or handles]

**Changes Made:**
- Bullet point list of changes

**Testing Notes:**
- How changes were tested
- Test results
- Edge cases considered

**Technical Details:**
- Implementation specifics
- Performance implications
- Dependencies added/updated

**Impact:**
- How this affects users
- Breaking changes (if any)
- Migration notes (if needed)
```

---

## Change Categories

- **Added**: New features, files, or functionality
- **Changed**: Changes to existing functionality
- **Deprecated**: Features that will be removed soon
- **Removed**: Removed features or files
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes
- **Performance**: Performance improvements
- **Documentation**: Documentation updates

---

**Notes:**
- Always update this file when making significant changes
- Include date, component, and brief description
- Link to related issues or PRs when applicable
- Be clear and concise
- Focus on what changed and why, not how
