"""
Deep Learning Helper Functions

Utilities for training and evaluating neural networks with PyTorch.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable


def calculate_conv_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1
) -> int:
    """
    Calculate output size of a convolutional layer.
    
    Parameters
    ----------
    input_size : int
        Input spatial dimension (height or width)
    kernel_size : int
        Size of the convolution kernel
    stride : int, default=1
        Stride of the convolution
    padding : int, default=0
        Padding added to input
    dilation : int, default=1
        Dilation factor for the kernel
    
    Returns
    -------
    int
        Output spatial dimension
    """
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def calculate_pool_output_size(
    input_size: int,
    kernel_size: int,
    stride: Optional[int] = None,
    padding: int = 0
) -> int:
    """
    Calculate output size of a pooling layer.
    
    Parameters
    ----------
    input_size : int
        Input spatial dimension
    kernel_size : int
        Size of the pooling kernel
    stride : int, optional
        Stride of pooling (defaults to kernel_size)
    padding : int, default=0
        Padding added to input
    
    Returns
    -------
    int
        Output spatial dimension
    """
    if stride is None:
        stride = kernel_size
    return (input_size + 2 * padding - kernel_size) // stride + 1


def count_parameters(model) -> int:
    """
    Count total trainable parameters in a model.
    
    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model
    
    Returns
    -------
    int
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_layer_output_shapes(model, input_shape: Tuple[int, ...]) -> List[Tuple[str, Tuple]]:
    """
    Get output shapes for each layer in a model.
    
    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model
    input_shape : tuple
        Shape of input tensor (without batch dimension)
    
    Returns
    -------
    list
        List of (layer_name, output_shape) tuples
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for this function")
    
    shapes = []
    hooks = []
    
    def hook_fn(name):
        def fn(module, input, output):
            shapes.append((name, tuple(output.shape)))
        return fn
    
    for name, layer in model.named_modules():
        if name:  # Skip the root module
            hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, *input_shape)
        model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return shapes


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Parameters
    ----------
    patience : int, default=7
        Number of epochs to wait for improvement
    min_delta : float, default=0
        Minimum change to qualify as improvement
    mode : str, default='min'
        'min' for loss, 'max' for metrics like accuracy
    restore_best_weights : bool, default=True
        Whether to restore model to best weights when stopping
    
    Attributes
    ----------
    best_score : float
        Best score achieved
    counter : int
        Counter for epochs without improvement
    early_stop : bool
        Whether to stop training
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
        if mode == 'min':
            self.compare = lambda current, best: current < best - min_delta
        else:
            self.compare = lambda current, best: current > best + min_delta
    
    def __call__(self, score: float, model=None) -> bool:
        """
        Check if training should stop.
        
        Parameters
        ----------
        score : float
            Current validation score
        model : torch.nn.Module, optional
            Model to save weights from
        
        Returns
        -------
        bool
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if model is not None and self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
        
        return self.early_stop
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None


class TrainingHistory:
    """
    Track training metrics over epochs.
    
    Attributes
    ----------
    history : dict
        Dictionary of metric lists
    """
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {}
    
    def update(self, metrics: Dict[str, float]):
        """
        Add metrics for an epoch.
        
        Parameters
        ----------
        metrics : dict
            Dictionary of metric name to value
        """
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get(self, metric: str) -> List[float]:
        """Get values for a metric."""
        return self.history.get(metric, [])
    
    def plot(self, metrics: Optional[List[str]] = None, figsize: Tuple[int, int] = (12, 4)):
        """
        Plot training curves.
        
        Parameters
        ----------
        metrics : list, optional
            List of metrics to plot (plots all if None)
        figsize : tuple
            Figure size
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
        
        if metrics is None:
            metrics = list(self.history.keys())
        
        n_plots = len(metrics)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        if n_plots == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            values = self.history.get(metric, [])
            ax.plot(values)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_title(metric)
        
        plt.tight_layout()
        return fig


def create_learning_rate_schedule(
    optimizer,
    schedule_type: str = 'cosine',
    epochs: int = 100,
    **kwargs
):
    """
    Create a learning rate scheduler.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        PyTorch optimizer
    schedule_type : str
        Type of schedule: 'step', 'cosine', 'exponential', 'plateau'
    epochs : int
        Total number of training epochs
    **kwargs
        Additional arguments for the scheduler
    
    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler
    """
    try:
        import torch.optim.lr_scheduler as lr_scheduler
    except ImportError:
        raise ImportError("PyTorch is required for this function")
    
    if schedule_type == 'step':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', epochs // 3),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif schedule_type == 'cosine':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=kwargs.get('eta_min', 0)
        )
    elif schedule_type == 'exponential':
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    elif schedule_type == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10)
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Compute class weights for imbalanced datasets.
    
    Parameters
    ----------
    labels : np.ndarray
        Array of class labels
    
    Returns
    -------
    np.ndarray
        Array of class weights
    """
    classes, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(classes)
    
    weights = n_samples / (n_classes * counts)
    return weights


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate classification accuracy.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels or logits
    targets : np.ndarray
        True class labels
    
    Returns
    -------
    float
        Accuracy score between 0 and 1
    """
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    return np.mean(predictions == targets)


def get_activation_function(name: str) -> Callable:
    """
    Get activation function by name.
    
    Parameters
    ----------
    name : str
        Name of activation function
    
    Returns
    -------
    callable
        Activation function
    """
    try:
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required for this function")
    
    activations = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'gelu': nn.GELU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'softmax': nn.Softmax
    }
    
    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    
    return activations[name_lower]
