# 🤖 Python Machine Learning Study Guide

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A comprehensive, hands-on curriculum for mastering Machine Learning with Python**

*From mathematical foundations to production deployment*

[Getting Started](#-getting-started) •
[Learning Path](#-learning-path) •
[Architecture](#-system-architecture) •
[Technologies](#-technology-stack)

</div>

---

## 📖 Table of Contents

- [Why This Project?](#-why-this-project)
- [Project Goals](#-project-goals)
- [System Architecture](#-system-architecture)
- [Learning Path](#-learning-path)
- [Technology Stack](#-technology-stack)
- [Project Timeline](#-project-timeline)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Current Progress](#-current-progress)

---

## 🎯 Why This Project?

### The Problem

Learning Machine Learning is challenging due to:

| Challenge | Impact | Our Solution |
|-----------|--------|--------------|
| **Fragmented Resources** | Learners jump between tutorials without cohesion | Unified, progressive curriculum |
| **Theory-Practice Gap** | Math concepts don't connect to code | Every concept has implementation |
| **No Production Focus** | Tutorials don't cover real-world deployment | End-to-end projects with deployment |
| **Outdated Content** | Many resources use deprecated libraries | Modern stack (PyTorch 2.0+, Python 3.12) |
| **Missing Testing** | No emphasis on code quality | TDD approach with pytest |

### The Solution

This study guide provides a **structured, progressive path** from Python basics to deploying production ML systems:

```mermaid
flowchart LR
    subgraph Foundation["🔧 Foundation"]
        A[NumPy]
        B[Pandas]
        C[Visualization]
    end
    
    subgraph Classical["📊 Classical ML"]
        D[Supervised]
        E[Unsupervised]
    end
    
    subgraph Deep["🧠 Deep Learning"]
        F[Neural Nets]
        G[CNNs]
        H[RNNs]
    end
    
    subgraph Applied["🚀 Applied"]
        I[NLP]
        J[Computer Vision]
        K[Projects]
    end
    
    Foundation --> Classical
    Classical --> Deep
    Deep --> Applied
    
    style Foundation fill:#1a1a2e,stroke:#00d4ff,color:#fff
    style Classical fill:#1a1a2e,stroke:#00ff88,color:#fff
    style Deep fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style Applied fill:#1a1a2e,stroke:#ffd93d,color:#fff
```

---

## 🎓 Project Goals

### Learning Objectives

```mermaid
mindmap
  root((ML Mastery))
    Foundations
      NumPy Arrays
      Pandas DataFrames
      Data Visualization
      Statistics
    Classical ML
      Regression
      Classification
      Clustering
      Dimensionality Reduction
    Deep Learning
      Neural Networks
      CNNs
      RNNs/LSTMs
      Transformers
    Applications
      NLP
      Computer Vision
      Time Series
      Deployment
```

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Notebooks Completed** | 50+ | Interactive Jupyter notebooks |
| **Unit Tests** | 90%+ coverage | pytest with coverage reports |
| **Projects Built** | 5+ end-to-end | From data to deployment |
| **Code Quality** | 100% type-hinted | mypy + pylint passing |

---

## 🏗️ System Architecture

### High-Level Overview

```mermaid
flowchart TB
    subgraph Input["📥 Input Layer"]
        direction TB
        NB["📓 Jupyter Notebooks"]
        DATA["📁 Datasets"]
        CFG["⚙️ Configs"]
    end
    
    subgraph Core["⚙️ Core Processing"]
        direction TB
        SRC["🐍 Source Code"]
        UTILS["🔧 Utilities"]
        MODELS["🤖 Models"]
        VIZ["📊 Visualization"]
    end
    
    subgraph Quality["✅ Quality Assurance"]
        direction TB
        TESTS["🧪 Tests"]
        LINT["📝 Linting"]
        DOCS["📚 Documentation"]
    end
    
    subgraph Deploy["�� Deployment"]
        direction TB
        DOCKER["🐳 Docker"]
        API["🌐 API"]
    end
    
    Input --> Core
    Core --> Quality
    Quality --> Deploy
    
    style Input fill:#2d3436,stroke:#00cec9,color:#fff
    style Core fill:#2d3436,stroke:#6c5ce7,color:#fff
    style Quality fill:#2d3436,stroke:#00b894,color:#fff
    style Deploy fill:#2d3436,stroke:#e17055,color:#fff
```

### Directory Architecture

```mermaid
flowchart TD
    subgraph Root["📁 python-ML-learn"]
        direction TB
        
        subgraph Learning["📚 Learning Modules"]
            F01["01-fundamentals/"]
            F02["02-supervised-learning/"]
            F03["03-unsupervised-learning/"]
            F04["04-deep-learning/"]
            F05["05-nlp/"]
            F06["06-computer-vision/"]
            F07["07-projects/"]
        end
        
        subgraph Source["💻 Source Code"]
            SRC_UTILS["src/utils/"]
            SRC_MODELS["src/models/"]
            SRC_DATA["src/data_processing/"]
            SRC_VIZ["src/visualization/"]
        end
        
        subgraph Support["🔧 Support"]
            TESTS["tests/"]
            DOCS["docs/"]
            DOCKER["docker/"]
            MEMORY["memory-bank/"]
        end
    end
    
    style Root fill:#1e272e,stroke:#fff,color:#fff
    style Learning fill:#2d3436,stroke:#74b9ff,color:#fff
    style Source fill:#2d3436,stroke:#a29bfe,color:#fff
    style Support fill:#2d3436,stroke:#55efc4,color:#fff
```

### Data Flow Architecture

```mermaid
flowchart LR
    subgraph Data["📊 Data Pipeline"]
        RAW["Raw Data"]
        CLEAN["Cleaned Data"]
        FEAT["Features"]
    end
    
    subgraph Model["🤖 Model Pipeline"]
        TRAIN["Training"]
        VAL["Validation"]
        TEST["Testing"]
    end
    
    subgraph Output["📈 Output"]
        PRED["Predictions"]
        METRICS["Metrics"]
        VIZ["Visualizations"]
    end
    
    RAW --> CLEAN
    CLEAN --> FEAT
    FEAT --> TRAIN
    TRAIN --> VAL
    VAL --> TEST
    TEST --> PRED
    TEST --> METRICS
    METRICS --> VIZ
    
    style Data fill:#2d3436,stroke:#00cec9,color:#fff
    style Model fill:#2d3436,stroke:#6c5ce7,color:#fff
    style Output fill:#2d3436,stroke:#fdcb6e,color:#fff
```

---

## 📚 Learning Path

### Phase Overview

```mermaid
flowchart TB
    subgraph P1["Phase 1: Foundation"]
        direction LR
        P1A["Week 1-2"]
        P1B["Infrastructure<br/>& Setup"]
        P1A --> P1B
    end
    
    subgraph P2["Phase 2: Fundamentals"]
        direction LR
        P2A["Week 3-5"]
        P2B["NumPy, Pandas<br/>Statistics, Viz"]
        P2A --> P2B
    end
    
    subgraph P3["Phase 3: Supervised"]
        direction LR
        P3A["Week 6-8"]
        P3B["Regression<br/>Classification"]
        P3A --> P3B
    end
    
    subgraph P4["Phase 4: Unsupervised"]
        direction LR
        P4A["Week 9-10"]
        P4B["Clustering<br/>PCA, t-SNE"]
        P4A --> P4B
    end
    
    subgraph P5["Phase 5: Deep Learning"]
        direction LR
        P5A["Week 11-13"]
        P5B["Neural Nets<br/>CNN, RNN"]
        P5A --> P5B
    end
    
    subgraph P6["Phase 6-9: Advanced"]
        direction LR
        P6A["Week 14-26"]
        P6B["NLP, CV<br/>Projects, MLOps"]
        P6A --> P6B
    end
    
    P1 --> P2 --> P3 --> P4 --> P5 --> P6
    
    style P1 fill:#1e3a5f,stroke:#3498db,color:#fff
    style P2 fill:#1e3a5f,stroke:#2ecc71,color:#fff
    style P3 fill:#1e3a5f,stroke:#9b59b6,color:#fff
    style P4 fill:#1e3a5f,stroke:#e74c3c,color:#fff
    style P5 fill:#1e3a5f,stroke:#f39c12,color:#fff
    style P6 fill:#1e3a5f,stroke:#1abc9c,color:#fff
```

### Curriculum Details

<details>
<summary><b>📘 Phase 1: Foundation (Weeks 1-2)</b></summary>

| Topic | Description | Deliverable |
|-------|-------------|-------------|
| Project Structure | Modular src layout | Folder hierarchy |
| Development Environment | VS Code + extensions | `.vscode/settings.json` |
| Docker Setup | Reproducible environment | `Dockerfile`, `docker-compose.yml` |
| Testing Framework | pytest configuration | `conftest.py`, `pytest.ini` |

**Status**: ✅ Complete

</details>

<details>
<summary><b>📗 Phase 2: Core ML Fundamentals (Weeks 3-5)</b></summary>

| Topic | Key Concepts | Notebook |
|-------|--------------|----------|
| **NumPy** | Arrays, broadcasting, linear algebra | `01_numpy_fundamentals.ipynb` |
| **Pandas** | DataFrames, cleaning, aggregation | `02_pandas_data_manipulation.ipynb` |
| **Visualization** | matplotlib, seaborn, plotly | `03_data_visualization.ipynb` |
| **Statistics** | Distributions, hypothesis testing | `04_statistics_probability.ipynb` |
| **Feature Engineering** | Scaling, encoding, pipelines | `05_feature_engineering.ipynb` |

**Status**: 🟡 In Progress (NumPy complete)

</details>

<details>
<summary><b>📙 Phase 3: Supervised Learning (Weeks 6-8)</b></summary>

| Algorithm | Mathematical Foundation | Implementation |
|-----------|------------------------|----------------|
| **Linear Regression** | $\hat{y} = X\beta$, MSE loss | From scratch + sklearn |
| **Logistic Regression** | Sigmoid, cross-entropy | Binary & multiclass |
| **Decision Trees** | Gini impurity, entropy | Visualization included |
| **Random Forests** | Bagging, feature importance | Hyperparameter tuning |
| **SVM** | Kernel trick, margin maximization | Multiple kernels |
| **Gradient Boosting** | Sequential ensembles | XGBoost, LightGBM |

**Status**: ⭕ Not Started

</details>

<details>
<summary><b>📕 Phase 4-9: Advanced Topics (Weeks 9-26)</b></summary>

| Phase | Topics | Hours |
|-------|--------|-------|
| **4. Unsupervised** | K-Means, DBSCAN, PCA, t-SNE | 50 |
| **5. Deep Learning** | Neural nets, CNN, RNN, PyTorch | 90 |
| **6. NLP** | Embeddings, BERT, Transformers | 70 |
| **7. Computer Vision** | Object detection, segmentation | 70 |
| **8. Projects** | End-to-end ML systems | 100+ |
| **9. MLOps** | Deployment, monitoring, CI/CD | 40 |

</details>

---

## 🛠️ Technology Stack

### Core Technologies Explained

```mermaid
flowchart TB
    subgraph Languages["🐍 Languages & Runtime"]
        PY["Python 3.12+"]
        JUP["Jupyter"]
    end
    
    subgraph DataScience["📊 Data Science"]
        NP["NumPy"]
        PD["Pandas"]
        SP["SciPy"]
    end
    
    subgraph Visualization["📈 Visualization"]
        MPL["Matplotlib"]
        SNS["Seaborn"]
        PLT["Plotly"]
    end
    
    subgraph ML["🤖 Machine Learning"]
        SK["scikit-learn"]
        XG["XGBoost"]
        LG["LightGBM"]
    end
    
    subgraph DL["🧠 Deep Learning"]
        PT["PyTorch"]
        TF["TensorFlow"]
        HF["Transformers"]
    end
    
    subgraph DevOps["🔧 DevOps"]
        DOC["Docker"]
        GIT["Git"]
        TEST["pytest"]
    end
    
    Languages --> DataScience
    Languages --> Visualization
    DataScience --> ML
    ML --> DL
    DL --> DevOps
    
    style Languages fill:#2c3e50,stroke:#3498db,color:#fff
    style DataScience fill:#2c3e50,stroke:#2ecc71,color:#fff
    style Visualization fill:#2c3e50,stroke:#9b59b6,color:#fff
    style ML fill:#2c3e50,stroke:#e74c3c,color:#fff
    style DL fill:#2c3e50,stroke:#f39c12,color:#fff
    style DevOps fill:#2c3e50,stroke:#1abc9c,color:#fff
```

### Technology Reference

| Technology | Version | Purpose | Why Chosen |
|------------|---------|---------|------------|
| **Python** | 3.12+ | Core language | Industry standard, rich ecosystem |
| **NumPy** | 2.4+ | Numerical computing | 10-100x faster than pure Python, vectorization |
| **Pandas** | 2.0+ | Data manipulation | Intuitive DataFrame API, SQL-like operations |
| **scikit-learn** | 1.3+ | Classical ML | Consistent API, comprehensive algorithms |
| **PyTorch** | 2.0+ | Deep learning | Dynamic graphs, Pythonic, research-friendly |
| **TensorFlow** | 2.13+ | Deep learning | Production-ready, TensorBoard, Keras API |
| **Matplotlib** | 3.7+ | Plotting | Highly customizable, publication quality |
| **Seaborn** | 0.12+ | Statistical viz | Beautiful defaults, statistical plots |
| **Docker** | Latest | Containerization | Reproducible environments |
| **pytest** | 7.4+ | Testing | Simple syntax, powerful fixtures |

### NumPy: The Foundation

```mermaid
flowchart LR
    subgraph NumPy["NumPy Ecosystem"]
        ARR["ndarray<br/>N-dimensional arrays"]
        UFUNC["ufuncs<br/>Element-wise ops"]
        LINALG["linalg<br/>Matrix operations"]
        RAND["random<br/>Statistical sampling"]
    end
    
    subgraph Benefits["Why NumPy?"]
        SPEED["⚡ 10-100x Faster"]
        MEM["💾 Memory Efficient"]
        BROAD["📡 Broadcasting"]
        INTER["🔗 Interoperability"]
    end
    
    NumPy --> Benefits
    
    style NumPy fill:#2c3e50,stroke:#013243,color:#fff
    style Benefits fill:#2c3e50,stroke:#4dabf7,color:#fff
```

**Definition**: NumPy is the fundamental package for scientific computing in Python.

**Motivation**: Python lists are slow for numerical operations. NumPy provides:
- Contiguous memory allocation
- Vectorized operations (no Python loops)
- C-level execution speed

**Mechanism**:
```python
# Python list (slow)
result = [x ** 2 for x in range(1000000)]  # ~200ms

# NumPy array (fast)
arr = np.arange(1000000)
result = arr ** 2  # ~2ms (100x faster!)
```

**Impact**: Enables processing of large datasets that would be impractical with pure Python.

---

## 📅 Project Timeline

### Gantt Chart

```mermaid
gantt
    title ML Study Guide - 26 Week Timeline
    dateFormat  YYYY-MM-DD
    
    section Phase 1
    Infrastructure Setup    :done,    p1, 2025-12-16, 2w
    
    section Phase 2
    NumPy Fundamentals      :done,    p2a, after p1, 3d
    Pandas & Data           :active,  p2b, after p2a, 1w
    Visualization           :         p2c, after p2b, 5d
    Statistics              :         p2d, after p2c, 4d
    Feature Engineering     :         p2e, after p2d, 5d
    
    section Phase 3
    Linear Regression       :         p3a, after p2e, 5d
    Logistic Regression     :         p3b, after p3a, 5d
    Decision Trees          :         p3c, after p3b, 6d
    SVM & Boosting          :         p3d, after p3c, 1w
    
    section Phase 4
    Clustering              :         p4a, after p3d, 1w
    Dimensionality Reduction:         p4b, after p4a, 1w
    
    section Phase 5
    Neural Networks         :         p5a, after p4b, 2w
    CNN & RNN               :         p5b, after p5a, 2w
    
    section Phase 6-9
    NLP                     :         p6, after p5b, 3w
    Computer Vision         :         p7, after p6, 3w
    Projects                :         p8, after p7, 4w
    MLOps                   :         p9, after p8, 2w
```

### Milestone Tracker

| Milestone | Target | Status | Progress |
|-----------|--------|--------|----------|
| M1: Infrastructure | Week 2 | ✅ Complete | ████████████ 100% |
| M2: Fundamentals | Week 5 | 🟡 In Progress | ██░░░░░░░░░░ 20% |
| M3: Supervised | Week 8 | ⭕ Not Started | ░░░░░░░░░░░░ 0% |
| M4: Unsupervised | Week 10 | ⭕ Not Started | ░░░░░░░░░░░░ 0% |
| M5: Deep Learning | Week 13 | ⭕ Not Started | ░░░░░░░░░░░░ 0% |
| M6: NLP | Week 16 | ⭕ Not Started | ░░░░░░░░░░░░ 0% |
| M7: Computer Vision | Week 19 | ⭕ Not Started | ░░░░░░░░░░░░ 0% |
| M8: Projects | Week 24 | ⭕ Not Started | ░░░░░░░░░░░░ 0% |
| M9: MLOps | Week 26 | ⭕ Not Started | ░░░░░░░░░░░░ 0% |

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.8+ | `python --version` |
| pip | Latest | `pip --version` |
| Git | Latest | `git --version` |
| Docker (optional) | Latest | `docker --version` |

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/python-ML-learn.git
cd python-ML-learn

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Jupyter Lab
jupyter lab
```

### Docker Setup (Recommended)

```bash
# Build and run with Docker Compose
cd docker
docker-compose up -d

# Access Jupyter Lab at http://localhost:8888
```

### Verify Installation

```bash
# Run tests to verify setup
python -m pytest tests/ -v

# Expected output: All tests passing
```

---

## 📁 Project Structure

```
python-ML-learn/
├── 📓 01-fundamentals/          # NumPy, Pandas, Visualization
│   ├── 01_numpy_fundamentals.ipynb
│   └── ...
├── 📓 02-supervised-learning/   # Regression, Classification
├── 📓 03-unsupervised-learning/ # Clustering, PCA
├── 📓 04-deep-learning/         # Neural Networks, CNN, RNN
├── 📓 05-nlp/                   # Text Processing, Transformers
├── 📓 06-computer-vision/       # Object Detection, Segmentation
├── 📓 07-projects/              # End-to-End Projects
│
├── 💻 src/                      # Source Code
│   ├── utils/                   # Utility functions
│   │   ├── timer.py            # Performance timing
│   │   ├── numpy_helpers.py    # NumPy utilities
│   │   └── ...
│   ├── models/                  # ML model implementations
│   ├── data_processing/         # Data pipelines
│   └── visualization/           # Plotting utilities
│
├── 🧪 tests/                    # Test Suite
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
│
├── 📚 docs/                     # Documentation
│   └── project-plan.md         # Detailed project plan
│
├── 🗃️ memory-bank/              # Project Memory
│   ├── change-log.md           # Version history
│   └── architecture-decisions/ # ADRs
│
├── 🐳 docker/                   # Docker Configuration
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── 📊 data/                     # Datasets
│   ├── raw/                    # Original data
│   └── processed/              # Cleaned data
│
├── ⚙️ configs/                   # Configuration files
├── 📜 requirements.txt          # Python dependencies
└── 📖 README.md                 # This file
```

---

## �� Current Progress

### Phase Completion

```mermaid
pie title Project Completion by Phase
    "Phase 1 - Complete" : 100
    "Phase 2 - In Progress" : 20
    "Phase 3-9 - Pending" : 0
```

### Test Coverage

| Module | Tests | Coverage | Status |
|--------|-------|----------|--------|
| `utils/timer.py` | 14 | 95% | ✅ |
| `utils/numpy_helpers.py` | 24 | 100% | ✅ |
| `utils/logger.py` | - | - | 🔄 |
| `utils/validators.py` | - | - | 🔄 |

**Total Tests**: 38 passing ✅

### Recent Updates

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-22 | v1.1.0 | NumPy fundamentals notebook, numpy_helpers module |
| 2025-12-22 | v1.0.0 | Initial project structure, Docker setup |

---

## 📈 Learning Tips

### Best Practices

1. **📐 Understand the Math**: Don't skip mathematical intuition
2. **💻 Code from Scratch**: Implement algorithms before using libraries
3. **📊 Visualize Everything**: Use plots to understand behavior
4. **�� Read Comments**: Code is heavily documented
5. **🔁 Practice Daily**: Consistency is key
6. **🧪 Write Tests**: Verify your implementations

### Study Schedule

```mermaid
flowchart LR
    subgraph Daily["Daily (2-3 hours)"]
        D1["📖 Theory<br/>30 min"]
        D2["💻 Coding<br/>90 min"]
        D3["📝 Review<br/>30 min"]
    end
    
    subgraph Weekly["Weekly"]
        W1["📓 1-2 Notebooks"]
        W2["🧪 Unit Tests"]
        W3["📊 Mini Project"]
    end
    
    Daily --> Weekly
    
    style Daily fill:#2c3e50,stroke:#3498db,color:#fff
    style Weekly fill:#2c3e50,stroke:#2ecc71,color:#fff
```

---

## 🔗 Resources

### Documentation

| Resource | Link | Description |
|----------|------|-------------|
| NumPy | [numpy.org](https://numpy.org/doc/) | Array computing |
| Pandas | [pandas.pydata.org](https://pandas.pydata.org/docs/) | Data analysis |
| scikit-learn | [scikit-learn.org](https://scikit-learn.org/stable/) | Machine learning |
| PyTorch | [pytorch.org](https://pytorch.org/docs/) | Deep learning |
| TensorFlow | [tensorflow.org](https://www.tensorflow.org/guide) | Deep learning |

### Learning Platforms

- �� [Kaggle Learn](https://www.kaggle.com/learn) - Free micro-courses
- 📊 [Papers With Code](https://paperswithcode.com/) - Research implementations
- 🎥 [3Blue1Brown](https://www.3blue1brown.com/) - Visual math explanations

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new code
4. Submit a pull request

---

## 📄 License

MIT License - Feel free to use for personal learning.

---

<div align="center">

**Made with ❤️ for Machine Learning Enthusiasts**

⭐ Star this repo if you find it helpful!

</div>
