"""
Unit tests for the deep_learning module.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ml_core.deep_learning import (
    calculate_conv_output_size,
    calculate_pool_output_size,
    count_parameters,
    EarlyStopping,
    TrainingHistory,
    compute_class_weights,
    accuracy
)


class TestCalculateConvOutputSize:
    """Tests for calculate_conv_output_size function."""
    
    def test_basic_conv(self):
        """Test basic convolution output size."""
        # 28x28 input, 3x3 kernel, no padding
        result = calculate_conv_output_size(28, 3, stride=1, padding=0)
        assert result == 26
    
    def test_conv_with_padding(self):
        """Test convolution with same padding."""
        # 28x28 input, 3x3 kernel, padding=1 -> same size
        result = calculate_conv_output_size(28, 3, stride=1, padding=1)
        assert result == 28
    
    def test_conv_with_stride(self):
        """Test convolution with stride."""
        # 28x28 input, 3x3 kernel, stride=2, padding=1
        result = calculate_conv_output_size(28, 3, stride=2, padding=1)
        assert result == 14
    
    def test_conv_with_dilation(self):
        """Test convolution with dilation."""
        result = calculate_conv_output_size(28, 3, stride=1, padding=0, dilation=2)
        assert result == 24
    
    def test_large_kernel(self):
        """Test with larger kernel."""
        result = calculate_conv_output_size(32, 5, stride=1, padding=2)
        assert result == 32


class TestCalculatePoolOutputSize:
    """Tests for calculate_pool_output_size function."""
    
    def test_basic_pool(self):
        """Test basic pooling output size."""
        # 28x28 input, 2x2 pool
        result = calculate_pool_output_size(28, 2)
        assert result == 14
    
    def test_pool_with_stride(self):
        """Test pooling with custom stride."""
        result = calculate_pool_output_size(28, 2, stride=2)
        assert result == 14
    
    def test_pool_with_padding(self):
        """Test pooling with padding."""
        result = calculate_pool_output_size(7, 2, stride=2, padding=1)
        assert result == 4
    
    def test_3x3_pool(self):
        """Test 3x3 pooling."""
        result = calculate_pool_output_size(32, 3, stride=2)
        assert result == 15


class TestCountParameters:
    """Tests for count_parameters function."""
    
    def test_linear_layer(self):
        """Test parameter counting for linear layer."""
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        
        model = nn.Linear(10, 5)
        # 10 * 5 weights + 5 biases = 55
        assert count_parameters(model) == 55
    
    def test_sequential_model(self):
        """Test parameter counting for sequential model."""
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )
        # (10*20 + 20) + (20*5 + 5) = 220 + 105 = 325
        assert count_parameters(model) == 325
    
    def test_frozen_parameters(self):
        """Test that frozen parameters are not counted."""
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        
        model = nn.Linear(10, 5)
        model.weight.requires_grad = False
        # Only bias is trainable = 5
        assert count_parameters(model) == 5


class TestEarlyStopping:
    """Tests for EarlyStopping class."""
    
    def test_no_early_stop_improving(self):
        """Test that early stopping doesn't trigger when improving."""
        es = EarlyStopping(patience=3)
        
        # Improving scores
        assert not es(1.0)
        assert not es(0.9)
        assert not es(0.8)
        assert not es(0.7)
        
        assert not es.early_stop
        assert es.counter == 0
    
    def test_early_stop_not_improving(self):
        """Test that early stopping triggers after patience."""
        es = EarlyStopping(patience=3)
        
        es(1.0)  # Best
        es(1.1)  # Worse, counter=1
        es(1.2)  # Worse, counter=2
        result = es(1.3)  # Worse, counter=3 -> stop
        
        assert result == True
        assert es.early_stop == True
    
    def test_counter_reset_on_improvement(self):
        """Test that counter resets when score improves."""
        es = EarlyStopping(patience=3)
        
        es(1.0)
        es(1.1)  # counter=1
        es(1.2)  # counter=2
        es(0.8)  # Improved! counter=0
        
        assert es.counter == 0
        assert es.best_score == 0.8
    
    def test_max_mode(self):
        """Test early stopping in max mode (e.g., accuracy)."""
        es = EarlyStopping(patience=2, mode='max')
        
        es(0.8)  # Best
        es(0.7)  # Worse
        result = es(0.6)  # Worse -> stop
        
        assert result == True
    
    def test_min_delta(self):
        """Test minimum delta for improvement."""
        es = EarlyStopping(patience=2, min_delta=0.1)
        
        es(1.0)  # Best
        es(0.95)  # Not enough improvement
        result = es(0.92)  # Still not enough -> stop
        
        assert result == True
    
    def test_reset(self):
        """Test reset functionality."""
        es = EarlyStopping(patience=2)
        
        es(1.0)
        es(1.1)
        es(1.2)  # Should trigger
        
        es.reset()
        
        assert es.counter == 0
        assert es.best_score is None
        assert es.early_stop == False


class TestTrainingHistory:
    """Tests for TrainingHistory class."""
    
    def test_update_single_metric(self):
        """Test updating single metric."""
        history = TrainingHistory()
        
        history.update({'loss': 1.0})
        history.update({'loss': 0.8})
        history.update({'loss': 0.6})
        
        assert history.get('loss') == [1.0, 0.8, 0.6]
    
    def test_update_multiple_metrics(self):
        """Test updating multiple metrics."""
        history = TrainingHistory()
        
        history.update({'loss': 1.0, 'accuracy': 0.5})
        history.update({'loss': 0.8, 'accuracy': 0.7})
        
        assert history.get('loss') == [1.0, 0.8]
        assert history.get('accuracy') == [0.5, 0.7]
    
    def test_get_nonexistent_metric(self):
        """Test getting a metric that doesn't exist."""
        history = TrainingHistory()
        
        assert history.get('nonexistent') == []


class TestComputeClassWeights:
    """Tests for compute_class_weights function."""
    
    def test_balanced_classes(self):
        """Test with balanced classes."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        weights = compute_class_weights(labels)
        
        # All classes have same count, weights should be equal
        np.testing.assert_array_almost_equal(weights, [1.0, 1.0, 1.0])
    
    def test_imbalanced_classes(self):
        """Test with imbalanced classes."""
        labels = np.array([0, 0, 0, 0, 1, 2])  # 4:1:1 ratio
        weights = compute_class_weights(labels)
        
        # Weight for class 0 should be lower
        assert weights[0] < weights[1]
        assert weights[0] < weights[2]
        assert weights[1] == weights[2]
    
    def test_binary_imbalanced(self):
        """Test binary classification with imbalance."""
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8:2 ratio
        weights = compute_class_weights(labels)
        
        # Weight for class 1 should be higher
        assert weights[1] > weights[0]


class TestAccuracy:
    """Tests for accuracy function."""
    
    def test_perfect_accuracy(self):
        """Test perfect predictions."""
        predictions = np.array([0, 1, 2, 0, 1])
        targets = np.array([0, 1, 2, 0, 1])
        
        assert accuracy(predictions, targets) == 1.0
    
    def test_zero_accuracy(self):
        """Test completely wrong predictions."""
        predictions = np.array([1, 0, 0, 1])
        targets = np.array([0, 1, 1, 0])
        
        assert accuracy(predictions, targets) == 0.0
    
    def test_partial_accuracy(self):
        """Test partial accuracy."""
        predictions = np.array([0, 1, 0, 1])
        targets = np.array([0, 0, 0, 1])
        
        assert accuracy(predictions, targets) == 0.75
    
    def test_accuracy_from_logits(self):
        """Test accuracy when given logits."""
        # Logits for 3 classes
        predictions = np.array([
            [2.0, 0.5, 0.1],  # argmax=0
            [0.1, 3.0, 0.5],  # argmax=1
            [0.5, 0.1, 2.5]   # argmax=2
        ])
        targets = np.array([0, 1, 2])
        
        assert accuracy(predictions, targets) == 1.0


class TestGetActivationFunction:
    """Tests for get_activation_function."""
    
    @pytest.mark.parametrize("name", [
        'relu', 'ReLU', 'RELU',
        'leaky_relu', 'elu', 'selu', 'gelu',
        'tanh', 'sigmoid', 'softmax'
    ])
    def test_valid_activations(self, name):
        """Test getting valid activation functions."""
        torch = pytest.importorskip("torch")
        from ml_core.deep_learning import get_activation_function
        
        activation = get_activation_function(name)
        assert activation is not None
    
    def test_invalid_activation(self):
        """Test invalid activation name raises error."""
        torch = pytest.importorskip("torch")
        from ml_core.deep_learning import get_activation_function
        
        with pytest.raises(ValueError):
            get_activation_function('invalid_activation')


class TestCreateLearningRateSchedule:
    """Tests for create_learning_rate_schedule function."""
    
    def test_step_schedule(self):
        """Test step learning rate schedule."""
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        import torch.optim as optim
        from ml_core.deep_learning import create_learning_rate_schedule
        
        model = nn.Linear(10, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        scheduler = create_learning_rate_schedule(optimizer, 'step', epochs=100)
        assert scheduler is not None
    
    def test_cosine_schedule(self):
        """Test cosine annealing schedule."""
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        import torch.optim as optim
        from ml_core.deep_learning import create_learning_rate_schedule
        
        model = nn.Linear(10, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        scheduler = create_learning_rate_schedule(optimizer, 'cosine', epochs=100)
        assert scheduler is not None
    
    def test_invalid_schedule(self):
        """Test invalid schedule type raises error."""
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        import torch.optim as optim
        from ml_core.deep_learning import create_learning_rate_schedule
        
        model = nn.Linear(10, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        with pytest.raises(ValueError):
            create_learning_rate_schedule(optimizer, 'invalid', epochs=100)
