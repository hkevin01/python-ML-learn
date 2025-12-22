# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.10.0] - 2025-07-08

### Added
- **Phase 6: Natural Language Processing Module**
  - Complete NLP curriculum with 5 notebooks
  - NLTK 3.9.2, transformers 4.57.3, datasets 4.4.2 installed

- **Text Preprocessing** (`05-nlp/01_text_preprocessing.ipynb`)
  - 18-cell comprehensive notebook
  - Text cleaning and normalization
  - Tokenization techniques (simple, NLTK, custom)
  - Stopword removal
  - Stemming (Porter algorithm)
  - Lemmatization with WordNet
  - Part-of-speech tagging
  - N-gram generation
  - TextPreprocessor class

- **Text Vectorization** (`05-nlp/02_text_vectorization.ipynb`)
  - 18-cell comprehensive notebook
  - Bag of Words (BoW) implementation
  - sklearn CountVectorizer
  - TF-IDF from scratch and with sklearn
  - Document similarity with cosine similarity
  - Latent Semantic Analysis (LSA)
  - One-hot encoding for words
  - Vectorization method comparison

- **Word Embeddings** (`05-nlp/03_word_embeddings.ipynb`)
  - 18-cell comprehensive notebook
  - Word embedding concepts
  - Skip-gram implementation from scratch
  - Negative sampling
  - PyTorch Embedding layers
  - Simple text classifier with embeddings
  - Word analogy tasks
  - Embedding visualization with PCA
  - Aggregation strategies (mean, max, weighted)

- **Text Classification** (`05-nlp/04_text_classification.ipynb`)
  - 18-cell comprehensive notebook
  - Naive Bayes classifier
  - Logistic Regression with TF-IDF
  - Linear SVM classifier
  - sklearn Pipeline for text classification
  - Neural network text classifier
  - Multi-class classification
  - Feature importance analysis
  - Confusion matrix visualization

- **Transformers Introduction** (`05-nlp/05_transformers_introduction.ipynb`)
  - 18-cell comprehensive notebook
  - Self-attention mechanism from scratch
  - Multi-head attention implementation
  - Positional encoding
  - Transformer encoder block
  - Hugging Face pipelines (sentiment, generation, QA, NER)
  - BERT tokenization
  - Pre-trained model usage
  - Sentence embeddings with mean pooling
  - Zero-shot classification

- **NLP Helper Module** (`src/ml_core/nlp.py`)
  - 14 helper functions for NLP tasks
  - clean_text, simple_tokenize, build_vocabulary
  - encode_texts, compute_tf, compute_idf, compute_tfidf
  - generate_ngrams, cosine_similarity
  - pad_sequences, create_attention_mask
  - TextPreprocessor class
  - get_word_frequencies, mean_pooling

- **NLP Unit Tests** (`tests/unit/test_nlp.py`)
  - 53 comprehensive unit tests
  - Tests for all NLP helper functions
  - Edge case coverage

### Changed
- Updated `src/ml_core/__init__.py` with NLP exports
- Total test count: 299 tests (all passing)

## [1.9.0] - 2025-07-08

### Added
- **Phase 5: Deep Learning Fundamentals Module**
  - Complete deep learning curriculum with 5 notebooks
  - PyTorch 2.9.1 installed as the deep learning framework

- **Neural Network Fundamentals** (`04-deep-learning/01_neural_network_fundamentals.ipynb`)
  - 17-cell comprehensive notebook
  - Perceptron visualization and implementation
  - Activation functions (sigmoid, ReLU, tanh, leaky ReLU)
  - Neural network from scratch with forward/backward propagation
  - Backpropagation algorithm explanation
  - Training on make_moons dataset
  - Architecture effects comparison
  - Learning rate effects analysis

- **PyTorch Introduction** (`04-deep-learning/02_pytorch_introduction.ipynb`)
  - 24-cell comprehensive notebook
  - Tensor creation and operations
  - Autograd and gradient computation
  - nn.Module and nn.Sequential
  - Loss functions and optimizers
  - DataLoader and Dataset
  - Complete training loop
  - Model saving and loading
  - BatchNorm and Dropout patterns

- **Convolutional Neural Networks** (`04-deep-learning/03_convolutional_neural_networks.ipynb`)
  - 19-cell comprehensive notebook
  - Manual convolution implementation
  - Edge detection kernels
  - PyTorch Conv2d and pooling layers
  - SimpleCNN architecture for MNIST
  - Training on MNIST dataset
  - Visualizing learned filters and feature maps
  - Making predictions with confidence

- **Recurrent Neural Networks** (`04-deep-learning/04_recurrent_neural_networks.ipynb`)
  - 18-cell comprehensive notebook
  - RNN cell from scratch
  - PyTorch RNN, LSTM, and GRU
  - LSTM gates visualization
  - Sequence prediction example (sine wave)
  - Bidirectional RNNs
  - Text classification architecture
  - RNN type comparison

- **Training Techniques** (`04-deep-learning/05_training_techniques.ipynb`)
  - 17-cell comprehensive notebook
  - Regularization: L1/L2, Dropout, BatchNorm
  - Optimizer comparison (SGD, Adam, RMSprop)
  - Learning rate schedules (Step, Cosine, OneCycle)
  - Early stopping implementation
  - Weight initialization methods
  - Gradient clipping
  - Mixed precision training concepts
  - Model checkpointing

- **Deep Learning Helpers** (`src/ml_core/deep_learning.py`)
  - `calculate_conv_output_size()` - conv layer output calculator
  - `calculate_pool_output_size()` - pooling layer output calculator
  - `count_parameters()` - model parameter counter
  - `get_layer_output_shapes()` - layer-by-layer shape tracking
  - `EarlyStopping` - early stopping with best weight restoration
  - `TrainingHistory` - training metrics tracker with plotting
  - `create_learning_rate_schedule()` - LR scheduler factory
  - `compute_class_weights()` - imbalanced dataset weights
  - `accuracy()` - classification accuracy metric
  - `get_activation_function()` - activation by name

- **Deep Learning Tests** (`tests/unit/test_deep_learning.py`)
  - 42 unit tests covering all deep learning helpers
  - Tests for conv/pool calculations, EarlyStopping, TrainingHistory
  - Tests for accuracy, class weights, schedulers, activations

### Changed
- Updated `src/ml_core/__init__.py` with deep learning module exports
- Total test count: 246 tests passing

## [1.8.0] - 2025-07-08

### Added
- **Phase 4: Unsupervised Learning Module**
  - Complete unsupervised learning curriculum with 3 notebooks

- **Clustering** (`03-unsupervised-learning/01_clustering.ipynb`)
  - 22-cell comprehensive notebook
  - K-Means from scratch implementation
  - Elbow method and silhouette analysis
  - Hierarchical clustering with dendrograms
  - DBSCAN for density-based clustering
  - Customer segmentation practical example

- **Dimensionality Reduction** (`03-unsupervised-learning/02_dimensionality_reduction.ipynb`)
  - 22-cell comprehensive notebook
  - Curse of dimensionality explanation
  - PCA from scratch and sklearn implementation
  - Variance explained analysis
  - Image reconstruction with PCA
  - t-SNE with perplexity tuning
  - UMAP for manifold learning
  - ML pipelines with dimensionality reduction

- **Anomaly Detection** (`03-unsupervised-learning/03_anomaly_detection.ipynb`)
  - 22-cell comprehensive notebook
  - Types of anomalies (point, contextual, collective)
  - Statistical methods (Z-score, IQR)
  - Isolation Forest algorithm
  - Local Outlier Factor (LOF)
  - One-Class SVM for novelty detection
  - Autoencoder concept for anomaly detection
  - Credit card fraud detection example
  - Algorithm selection guide

- **Unsupervised Learning Helpers** (`src/ml_core/unsupervised.py`)
  - `find_optimal_clusters()` - elbow and silhouette methods
  - `evaluate_clustering()` - silhouette, Davies-Bouldin, Calinski-Harabasz
  - `compare_clustering_algorithms()` - multi-algorithm comparison
  - `find_optimal_dbscan_params()` - DBSCAN parameter tuning
  - `get_pca_variance_analysis()` - variance explained analysis
  - `get_pca_loadings()` - feature contribution to components
  - `detect_anomalies_zscore()` - Z-score anomaly detection
  - `detect_anomalies_iqr()` - IQR anomaly detection
  - `get_cluster_summary()` - cluster statistics
  - `assign_cluster_to_new_data()` - assign new points to clusters

- **Unsupervised Learning Tests** (`tests/unit/test_unsupervised.py`)
  - 37 unit tests covering all unsupervised helpers
  - Tests for clustering, PCA, and anomaly detection functions

### Changed
- Updated `src/ml_core/__init__.py` with unsupervised module exports
- Total test count: 204 tests passing

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
