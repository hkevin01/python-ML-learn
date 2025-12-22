"""
Unsupervised Learning Helper Functions

This module provides utility functions for clustering, dimensionality reduction,
and anomaly detection algorithms.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def find_optimal_clusters(
    X: np.ndarray,
    k_range: Tuple[int, int] = (2, 11),
    method: str = 'elbow',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Find optimal number of clusters using elbow or silhouette method.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    k_range : Tuple[int, int]
        Range of k values to try (min, max). Default is (2, 11)
    method : str
        Method to use: 'elbow', 'silhouette', or 'both'. Default is 'elbow'
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'k_values': list of k values tested
        - 'inertias': inertia values for each k (if elbow)
        - 'silhouette_scores': silhouette scores for each k (if silhouette)
        - 'optimal_k': recommended number of clusters
        
    Example
    -------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=200, centers=4, random_state=42)
    >>> result = find_optimal_clusters(X, k_range=(2, 8), method='both')
    >>> print(f"Optimal k: {result['optimal_k']}")
    """
    k_values = list(range(k_range[0], k_range[1]))
    results = {'k_values': k_values}
    
    if method in ['elbow', 'both']:
        inertias = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        results['inertias'] = inertias
        
    if method in ['silhouette', 'both']:
        silhouette_scores = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        results['silhouette_scores'] = silhouette_scores
        results['optimal_k'] = k_values[np.argmax(silhouette_scores)]
    else:
        # For elbow only, use second derivative to find elbow point
        inertias = np.array(results['inertias'])
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        elbow_idx = np.argmax(second_diffs) + 1  # +1 because of diff offset
        results['optimal_k'] = k_values[elbow_idx]
        
    return results


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate clustering quality using multiple metrics.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Cluster labels for each sample
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'silhouette': Silhouette score (-1 to 1, higher is better)
        - 'davies_bouldin': Davies-Bouldin index (lower is better)
        - 'calinski_harabasz': Calinski-Harabasz index (higher is better)
        - 'n_clusters': Number of clusters found
        
    Example
    -------
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
    >>> labels = KMeans(n_clusters=3).fit_predict(X)
    >>> metrics = evaluate_clustering(X, labels)
    >>> print(f"Silhouette: {metrics['silhouette']:.3f}")
    """
    # Filter out noise points (label = -1) for some metrics
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters < 2 or mask.sum() < 2:
        return {
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan,
            'n_clusters': n_clusters
        }
    
    X_clean = X[mask]
    labels_clean = labels[mask]
    
    return {
        'silhouette': silhouette_score(X_clean, labels_clean),
        'davies_bouldin': davies_bouldin_score(X_clean, labels_clean),
        'calinski_harabasz': calinski_harabasz_score(X_clean, labels_clean),
        'n_clusters': n_clusters
    }


def compare_clustering_algorithms(
    X: np.ndarray,
    algorithms: Optional[Dict[str, Any]] = None,
    scale: bool = True
) -> pd.DataFrame:
    """
    Compare multiple clustering algorithms on the same dataset.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    algorithms : Dict[str, Any], optional
        Dictionary of algorithm name to fitted/unfitted sklearn estimator.
        If None, uses default algorithms.
    scale : bool
        Whether to standardize features before clustering
        
    Returns
    -------
    pd.DataFrame
        Comparison results with columns: Algorithm, Silhouette, 
        Davies-Bouldin, Calinski-Harabasz, N_Clusters
        
    Example
    -------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
    >>> results = compare_clustering_algorithms(X)
    >>> print(results)
    """
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    if algorithms is None:
        algorithms = {
            'KMeans (k=3)': KMeans(n_clusters=3, random_state=42, n_init=10),
            'KMeans (k=5)': KMeans(n_clusters=5, random_state=42, n_init=10),
            'Hierarchical (k=3)': AgglomerativeClustering(n_clusters=3),
            'DBSCAN (eps=0.5)': DBSCAN(eps=0.5, min_samples=5)
        }
    
    results = []
    for name, model in algorithms.items():
        labels = model.fit_predict(X)
        metrics = evaluate_clustering(X, labels)
        metrics['Algorithm'] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df[['Algorithm', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'n_clusters']]
    df.columns = ['Algorithm', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'N_Clusters']
    
    return df


def find_optimal_dbscan_params(
    X: np.ndarray,
    eps_range: Tuple[float, float] = (0.1, 2.0),
    min_samples_range: Tuple[int, int] = (3, 10),
    n_eps: int = 10,
    scale: bool = True
) -> Dict[str, Any]:
    """
    Find optimal DBSCAN parameters using grid search.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    eps_range : Tuple[float, float]
        Range of epsilon values to try
    min_samples_range : Tuple[int, int]
        Range of min_samples values to try
    n_eps : int
        Number of epsilon values to test
    scale : bool
        Whether to standardize features
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'best_eps': optimal epsilon value
        - 'best_min_samples': optimal min_samples value
        - 'best_silhouette': best silhouette score achieved
        - 'results_df': DataFrame with all results
        
    Example
    -------
    >>> from sklearn.datasets import make_moons
    >>> X, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
    >>> result = find_optimal_dbscan_params(X)
    >>> print(f"Best eps: {result['best_eps']:.2f}")
    """
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    eps_values = np.linspace(eps_range[0], eps_range[1], n_eps)
    min_samples_values = list(range(min_samples_range[0], min_samples_range[1] + 1))
    
    results = []
    best_score = -1
    best_params = {}
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = (labels == -1).sum() / len(labels)
            
            if n_clusters >= 2 and noise_ratio < 0.5:
                score = silhouette_score(X[labels != -1], labels[labels != -1])
            else:
                score = -1
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'noise_ratio': noise_ratio,
                'silhouette': score
            })
            
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
    
    return {
        'best_eps': best_params.get('eps'),
        'best_min_samples': best_params.get('min_samples'),
        'best_silhouette': best_score,
        'results_df': pd.DataFrame(results)
    }


def get_pca_variance_analysis(
    X: np.ndarray,
    n_components: Optional[int] = None,
    scale: bool = True
) -> Dict[str, Any]:
    """
    Analyze variance explained by PCA components.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    n_components : int, optional
        Number of components to analyze. If None, uses min(n_samples, n_features)
    scale : bool
        Whether to standardize features before PCA
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'explained_variance_ratio': variance ratio for each component
        - 'cumulative_variance': cumulative variance explained
        - 'n_components_95': number of components for 95% variance
        - 'n_components_99': number of components for 99% variance
        - 'pca': fitted PCA object
        
    Example
    -------
    >>> from sklearn.datasets import load_digits
    >>> X = load_digits().data
    >>> result = get_pca_variance_analysis(X)
    >>> print(f"Components for 95% variance: {result['n_components_95']}")
    """
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])
    
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    
    # Find components needed for thresholds
    n_95 = np.argmax(cumulative >= 0.95) + 1
    n_99 = np.argmax(cumulative >= 0.99) + 1
    
    return {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': cumulative,
        'n_components_95': n_95,
        'n_components_99': n_99,
        'pca': pca
    }


def get_pca_loadings(
    pca: PCA,
    feature_names: Optional[List[str]] = None,
    n_components: int = 3
) -> pd.DataFrame:
    """
    Get PCA loadings (feature contributions) as a DataFrame.
    
    Parameters
    ----------
    pca : PCA
        Fitted PCA object
    feature_names : List[str], optional
        Names of original features. If None, uses Feature_0, Feature_1, etc.
    n_components : int
        Number of principal components to show. Default is 3
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features as rows and principal components as columns,
        showing the loading (weight) of each feature on each component
        
    Example
    -------
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.datasets import load_iris
    >>> X = load_iris().data
    >>> feature_names = load_iris().feature_names
    >>> pca = PCA(n_components=4).fit(X)
    >>> loadings = get_pca_loadings(pca, feature_names)
    >>> print(loadings)
    """
    n_components = min(n_components, pca.n_components_)
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(pca.components_.shape[1])]
    
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        index=feature_names,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return loadings


def detect_anomalies_zscore(
    X: np.ndarray,
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect anomalies using Z-score method.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    threshold : float
        Z-score threshold. Points with |z| > threshold are anomalies.
        Default is 3.0
        
    Returns
    -------
    np.ndarray
        Boolean array where True indicates an anomaly
        
    Example
    -------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 2)
    >>> X[0] = [10, 10]  # Add outlier
    >>> anomalies = detect_anomalies_zscore(X)
    >>> print(f"Number of anomalies: {anomalies.sum()}")
    """
    from scipy import stats
    z_scores = np.abs(stats.zscore(X, axis=0))
    return np.any(z_scores > threshold, axis=1)


def detect_anomalies_iqr(
    X: np.ndarray,
    k: float = 1.5
) -> np.ndarray:
    """
    Detect anomalies using IQR (Interquartile Range) method.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    k : float
        IQR multiplier. Points outside [Q1 - k*IQR, Q3 + k*IQR] are anomalies.
        Default is 1.5
        
    Returns
    -------
    np.ndarray
        Boolean array where True indicates an anomaly
        
    Example
    -------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 2)
    >>> X[0] = [10, 10]  # Add outlier
    >>> anomalies = detect_anomalies_iqr(X)
    >>> print(f"Number of anomalies: {anomalies.sum()}")
    """
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    
    return np.any((X < lower) | (X > upper), axis=1)


def get_cluster_summary(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get summary statistics for each cluster.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    labels : np.ndarray
        Cluster labels for each sample
    feature_names : List[str], optional
        Names of features. If None, uses Feature_0, Feature_1, etc.
        
    Returns
    -------
    pd.DataFrame
        Summary statistics for each cluster including mean, std, and count
        
    Example
    -------
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
    >>> labels = KMeans(n_clusters=3).fit_predict(X)
    >>> summary = get_cluster_summary(X, labels)
    >>> print(summary)
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['Cluster'] = labels
    
    # Calculate statistics per cluster
    summary = df.groupby('Cluster').agg(['mean', 'std', 'count'])
    
    # Flatten column names
    summary.columns = [f'{col[0]}_{col[1]}' for col in summary.columns]
    
    # Add cluster size as first column
    summary.insert(0, 'Size', df.groupby('Cluster').size())
    
    return summary


def assign_cluster_to_new_data(
    X_new: np.ndarray,
    centroids: np.ndarray
) -> np.ndarray:
    """
    Assign new data points to clusters based on centroids.
    
    Parameters
    ----------
    X_new : np.ndarray
        New data points of shape (n_samples, n_features)
    centroids : np.ndarray
        Cluster centroids of shape (n_clusters, n_features)
        
    Returns
    -------
    np.ndarray
        Cluster labels for new data
        
    Example
    -------
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
    >>> kmeans = KMeans(n_clusters=3).fit(X)
    >>> X_new = [[0, 0], [5, 5]]
    >>> labels = assign_cluster_to_new_data(X_new, kmeans.cluster_centers_)
    """
    X_new = np.asarray(X_new)
    distances = np.linalg.norm(X_new[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)
