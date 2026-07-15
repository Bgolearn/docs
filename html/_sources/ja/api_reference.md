# API Reference

```{note}
本ページは Bgolearn マニュアルの日本語版です。
```

```{warning}
**Common Error**: `AttributeError: 'numpy.ndarray' object has no attribute 'columns'`

This error occurs when you pass numpy arrays instead of pandas DataFrames to Bgolearn methods.

**Solution**: Always use pandas DataFrames and Series:
- `data_matrix` → `pd.DataFrame`
- `Measured_response` → `pd.Series`
- `virtual_samples` → `pd.DataFrame`

**Never use `.values`** - it converts DataFrames to numpy arrays!
```

## Core Classes

### BGOsampling.Bgolearn

The main optimization class that provides Bayesian Global Optimization functionality.

```python
from Bgolearn import BGOsampling

class Bgolearn:
    """
    Bayesian Global Optimization for materials discovery and scientific research.

    This class implements various acquisition functions and provides tools for
    efficient experimental design and optimization.
    """
```

#### Constructor

```python
def __init__(self):
    """
    Initialize Bgolearn optimizer.

    Creates necessary directories and displays welcome information.
    The optimizer is configured with default settings and can be customized
    through the fit() method parameters.
    """
```

#### Core Methods

##### fit()

```python
def fit(self, data_matrix, Measured_response, virtual_samples,
        Mission='Regression', Classifier='GaussianProcess', noise_std=None,
        Kriging_model=None, opt_num=1, min_search=True, CV_test=False,
        Dynamic_W=False, seed=42, Normalize=True):
    """
    Fit the Bayesian optimization model and recommend next experiments.

    Parameters
    ----------
    data_matrix : pandas.DataFrame
        Training input features (n_samples, n_features)
        **Must be DataFrame** - numpy arrays will cause AttributeError
    Measured_response : pandas.Series
        Training target values (n_samples,)
        **Must be Series** - numpy arrays may cause issues
    virtual_samples : pandas.DataFrame
        Candidate points for optimization (n_candidates, n_features)
        **Must be DataFrame** - numpy arrays will cause AttributeError
    Mission : str, default='Regression'
        Mission type: 'Regression' or 'Classification'
    Classifier : str, default='GaussianProcess'
        For classification: 'GaussianProcess', 'LogisticRegression',
        'NaiveBayes', 'SVM', 'RandomForest'
    noise_std : float or ndarray, default=None
        Noise level for Gaussian Process. If None, automatically estimated
    Kriging_model : str or callable, default=None
        Surrogate model: 'SVM', 'RF', 'AdaB', 'MLP' or custom model
    opt_num : int, default=1
        Number of recommended candidates for next iteration
    min_search : bool, default=True
        True for minimization, False for maximization
    CV_test : False, int, or 'LOOCV', default=False
        Cross-validation settings:
        - False: No cross-validation (default)
        - int: k-fold cross-validation (e.g., 10 for 10-fold)
        - 'LOOCV': Leave-one-out cross-validation
    Dynamic_W : bool, default=False
        Whether to apply dynamic resampling
    seed : int, default=42
        Random seed for reproducibility
    Normalize : bool, default=True
        Whether to normalize input data

    Returns
    -------
    Global_min or Global_max : optimization model
        Optimization model object with methods and attributes:
        - EI(): Expected Improvement acquisition function
        - UCB(): Upper Confidence Bound acquisition function
        - PoI(): Probability of Improvement acquisition function
        - virtual_samples_mean: predicted values for all candidates
        - virtual_samples_std: prediction uncertainties
        - data_matrix: training features
        - Measured_response: training targets

    Examples
    --------
    >>> from Bgolearn import BGOsampling
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> X = pd.DataFrame(np.random.randn(20, 3), columns=['x1', 'x2', 'x3'])
    >>> y = pd.Series(np.random.randn(20))
    >>> candidates = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])
    >>>
    >>> # Fit model
    >>> optimizer = BGOsampling.Bgolearn()
    >>> model = optimizer.fit(
    ...     data_matrix=X,  # Pass DataFrame directly
    ...     Measured_response=y,  # Pass Series directly
    ...     virtual_samples=candidates,  # Pass DataFrame directly
    ...     opt_num=1,
    ...     min_search=True
    ... )
    >>>
    >>> # Get recommendation using Expected Improvement
    >>> ei_values, recommended_points = model.EI()
    >>> next_experiment = recommended_points[0]  # First recommendation
    """
```

## Available Acquisition Functions  (single target）

Bgolearn supports multiple acquisition functions for different optimization scenarios:

### Core Acquisition Functions

#### Expected Improvement (EI)
- **Standard EI**: Basic expected improvement
- **Logarithmic EI**: Applies a logarithmic transform to EI 
- **EI_plugin**: Plugin-based expected improvement
- **Augmented_EI**: Enhanced EI with additional parameters
- **EQI**: Expected Quantile Improvement
- **Reinterpolation_EI**: Reinterpolation-based EI

#### Exploration-Based Functions
- **UCB**: Upper Confidence Bound
- **PoI**: Probability of Improvement

#### Information-Based Functions
- **PES**: Predictive Entropy Search
- **Knowledge_G**: Knowledge Gradient

### Usage in fit() Method

The acquisition function is automatically selected and applied during the `fit()` method. The optimization process uses these functions internally to recommend the next experimental point(s).

```python
# Example: Basic optimization with automatic acquisition function selection
from Bgolearn import BGOsampling

optimizer = BGOsampling.Bgolearn()
model = optimizer.fit(
    data_matrix=X_train,  # Pass DataFrame directly
    Measured_response=y_train,  # Pass Series directly
    virtual_samples=X_candidates,  # Pass DataFrame directly
    opt_num=1,  # Number of recommendations
    min_search=True  # Minimize objective, min_search=False for maximization
)

# Get the recommended point using EI
ei_values, recommended_points = model.EI()
next_experiment = recommended_points[0]
```

### Advanced Usage with BGO_Efficient

For advanced users who want to explicitly control acquisition functions:

```python
from Bgolearn.BgolearnFuns.BGO_eval import BGO_Efficient

# Create BGO_Efficient instance (advanced usage)
# This requires more setup and is typically used for research purposes
    next_point : numpy.ndarray
        Coordinates of the point with maximum UCB
    
    Examples
    --------
    >>> # Conservative exploration
    >>> ucb_values, next_point = model.UCB(alpha=1.0)
    >>> 
    >>> # Aggressive exploration
    >>> ucb_values, next_point = model.UCB(alpha=3.0)
    """
```

#### Probability of Improvement (PoI)

```python
def PoI(self, tao=0.01, T=None):
    """
    Probability of Improvement acquisition function.
    
    Parameters
    ----------
    tao : float, default=0.01
        Tolerance parameter for improvement
    T : float, optional
        Threshold value. If None, uses best observed value.
    
    Returns
    -------
    poi_values : numpy.ndarray
        PoI values for all candidate points
    next_point : numpy.ndarray
        Coordinates of the point with maximum PoI
    
    Examples
    --------
    >>> # Strict improvement required
    >>> poi_values, next_point = model.PoI(tao=0.0)
    >>> 
    >>> # Allow small degradation
    >>> poi_values, next_point = model.PoI(tao=0.1)
    """
```

#### Augmented Expected Improvement (AEI)

```python
def Augmented_EI(self, alpha=1.0, tao=0.0):
    """
    Augmented Expected Improvement for noisy functions.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Trade-off coefficient for baseline selection
    tao : float, default=0.0
        Noise standard deviation estimate
    
    Returns
    -------
    aei_values : numpy.ndarray
        AEI values for all candidate points
    next_point : numpy.ndarray
        Coordinates of the point with maximum AEI
    
    Examples
    --------
    >>> # For noisy experiments
    >>> aei_values, next_point = model.Augmented_EI(alpha=1.5, tao=0.1)
    """
```

#### Expected Quantile Improvement (EQI)

```python
def EQI(self, beta=0.5):
    """
    Expected Quantile Improvement for robust optimization.
    
    Parameters
    ----------
    beta : float, default=0.5
        Quantile level (0 < beta < 1)
        0.5 = median, 0.25 = conservative, 0.75 = optimistic
    
    Returns
    -------
    eqi_values : numpy.ndarray
        EQI values for all candidate points
    next_point : numpy.ndarray
        Coordinates of the point with maximum EQI
    
    Examples
    --------
    >>> # Optimize median performance
    >>> eqi_values, next_point = model.EQI(beta=0.5)
    >>> 
    >>> # Conservative optimization (25th percentile)
    >>> eqi_values, next_point = model.EQI(beta=0.25)
    """
```

#### Predictive Entropy Search (PES)

```python
def PES(self, sam_num=100):
    """
    Predictive Entropy Search for information-theoretic optimization.
    
    Parameters
    ----------
    sam_num : int, default=100
        Number of Monte Carlo samples for entropy estimation
    
    Returns
    -------
    pes_values : numpy.ndarray
        PES values for all candidate points
    next_point : numpy.ndarray
        Coordinates of the point with maximum PES
    
    Examples
    --------
    >>> # Standard precision
    >>> pes_values, next_point = model.PES(sam_num=100)
    >>> 
    >>> # High precision (slower)
    >>> pes_values, next_point = model.PES(sam_num=500)
    """
```

#### Knowledge Gradient (KG)

```python
def Knowledge_G(self, MC_num=1, Proc_num=1):
    """
    Knowledge Gradient for value of information optimization.
    
    Parameters
    ----------
    MC_num : int, default=1
        Number of Monte Carlo samples
    Proc_num : int, default=1
        Number of parallel processes (if supported)
    
    Returns
    -------
    kg_values : numpy.ndarray
        KG values for all candidate points
    next_point : numpy.ndarray
        Coordinates of the point with maximum KG
    
    Examples
    --------
    >>> # Fast approximation
    >>> kg_values, next_point = model.Knowledge_G(MC_num=1)
    >>> 
    >>> # More accurate estimation
    >>> kg_values, next_point = model.Knowledge_G(MC_num=5)
    """
```



## Basic Visualization

Bgolearn does not include built-in visualization methods. However, you can easily create visualizations using matplotlib:

```python
import matplotlib.pyplot as plt

# Example: Plot Expected Improvement values
ei_values, recommended_points = model.EI()

plt.figure(figsize=(10, 6))
plt.plot(ei_values)
plt.title('Expected Improvement Values')
plt.xlabel('Candidate Index')
plt.ylabel('EI Value')
plt.axvline(x=np.argmax(ei_values), color='red', linestyle='--',
           label=f'Best EI (idx={np.argmax(ei_values)})')
plt.legend()
plt.grid(True)
plt.show()
```

```python
# Example: Plot predictions vs uncertainties
predictions = model.virtual_samples_mean
uncertainties = model.virtual_samples_std

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(predictions, uncertainties, alpha=0.6)
plt.xlabel('Predicted Value')
plt.ylabel('Prediction Uncertainty')
plt.title('Prediction vs Uncertainty')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(predictions, bins=20, alpha=0.7)
plt.xlabel('Predicted Value')
plt.ylabel('Frequency')
plt.title('Distribution of Predictions')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Notes on Visualization

```{note}
Bgolearn focuses on optimization algorithms rather than visualization. For advanced plotting:

1. **Use matplotlib** for basic plots (as shown above)
2. **Use seaborn** for statistical visualizations
3. **Use plotly** for interactive plots
4. **Use BgoFace GUI** for visual optimization workflows

The examples above show how to extract data from Bgolearn models and create custom visualizations.
```
        ----------
        figsize : tuple, default=(10, 6)
            Figure size (width, height) in inches
        dpi : int, default=100
            Figure resolution
        style : str, default='seaborn'
            Matplotlib style to use
        
        Examples
        --------
        >>> from bgolearn.visualization import BgolearnVisualizer
        >>> visualizer = BgolearnVisualizer(figsize=(12, 8), dpi=150)
        """
```

#### Visualization Methods

```python
def plot_optimization_history(self, y_history, y_true_optimum=None, 
                            title="Optimization History", save_path=None):
    """
    Plot optimization convergence history.
    
    Parameters
    ----------
    y_history : list or numpy.ndarray
        Best objective values over iterations
    y_true_optimum : float, optional
        True optimum value for comparison
    title : str, default="Optimization History"
        Plot title
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    
    Examples
    --------
    >>> history = [1.0, 0.8, 0.5, 0.3, 0.1]
    >>> fig = visualizer.plot_optimization_history(
    ...     y_history=history,
    ...     y_true_optimum=0.0,
    ...     title="My Optimization"
    ... )
    """
```

```python
def plot_acquisition_function_2d(self, X_candidates, acquisition_values, 
                               X_train, y_train, next_point=None,
                               title="2D Acquisition Function", save_path=None):
    """
    Plot 2D acquisition function heatmap.
    
    Parameters
    ----------
    X_candidates : numpy.ndarray
        Candidate points (n_candidates, 2)
    acquisition_values : numpy.ndarray
        Acquisition function values
    X_train : numpy.ndarray
        Training input points
    y_train : numpy.ndarray
        Training target values
    next_point : numpy.ndarray, optional
        Next recommended point to highlight
    title : str, default="2D Acquisition Function"
        Plot title
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    
    Examples
    --------
    >>> ei_values, next_point = model.EI()
    >>> fig = visualizer.plot_acquisition_function_2d(
    ...     X_candidates, ei_values,
    ...     X_train, y_train,
    ...     next_point=next_point
    ... )
    """
```

```python
def plot_pareto_front(self, objectives, labels=None, 
                     title="Pareto Front", save_path=None):
    """
    Plot Pareto front for multi-objective optimization.
    
    Parameters
    ----------
    objectives : numpy.ndarray
        Objective values (n_points, n_objectives)
    labels : list, optional
        Objective labels
    title : str, default="Pareto Front"
        Plot title
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    
    Examples
    --------
    >>> # For 2-objective optimization
    >>> objectives = np.column_stack([strength, ductility])
    >>> fig = visualizer.plot_pareto_front(
    ...     objectives, 
    ...     labels=['Strength', 'Ductility']
    ... )
    """
```





## Quick Reference

### Common Workflows

```python
# Basic optimization workflow
from Bgolearn import BGOsampling

# 1. Prepare data
optimizer = BGOsampling.Bgolearn()
model = optimizer.fit(X_train, y_train, X_candidates)

# 2. Single-point optimization
ei_values, next_point = model.EI()

```

### Parameter Guidelines

| Function | Key Parameters | Recommended Values |
|----------|---------------|-------------------|
| EI | T | None (auto) or custom threshold |
| UCB | alpha | 1.0-3.0 (higher = more exploration) |
| PoI | tao | 0.0-0.1 (tolerance for improvement) |
| AEI | alpha, tao | 1.0-2.0, 0.0-0.2 |
| EQI | beta | 0.25 (conservative), 0.5 (median), 0.75 (optimistic) |
| PES | sam_num | 100-500 (higher = more accurate) |
| KG | MC_num | 1-5 (higher = more accurate) |
