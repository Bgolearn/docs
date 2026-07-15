# API参考

```{note}
本页是 Bgolearn 手册的中文版本。
```

```{warning}
**常见错误**：`AttributeError: 'numpy.ndarray' object has no attribute 'columns'`

当您将 numpy 数组而不是 pandas DataFrame 传递给 Bgolearn 方法时，会发生此错误。

**解决方案**：始终使用 pandas DataFrames 和 Series：
- `data_matrix` → `pd.DataFrame`
- `Measured_response` → `pd.Series`
- `virtual_samples` → `pd.DataFrame`

**永远不要使用 `.values`** - 它将 DataFrame 转换为 numpy 数组！
```

## 核心课程

### BGOsampling.Bgolearn

提供贝叶斯全局优化功能的主要优化类。

```python
from Bgolearn import BGOsampling

class Bgolearn:
    """
    Bayesian Global Optimization for materials discovery and scientific research.

    This class implements various acquisition functions and provides tools for
    efficient experimental design and optimization.
    """
```

#### 构造函数

```python
def __init__(self):
    """
    Initialize Bgolearn optimizer.

    Creates necessary directories and displays welcome information.
    The optimizer is configured with default settings and can be customized
    through the fit() method parameters.
    """
```

#### 核心方法

##### 合身（）

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

## 可用的采集功能（单个目标）

Bgolearn支持多种采集功能，针对不同的优化场景：

### 核心采集功能

#### 预期改善 (EI)
- **标准EI**：基本预期改善
- **对数 EI**：对 EI 应用对数变换 
- **EI_plugin**：基于插件的预期改进
- **Augmented_EI**：带有附加参数的增强型 EI
- **EQI**：预期分位数改进
- **Reinterpolation_EI**：基于重新插值的 EI

#### 基于探索的功能
- **UCB**：置信上限
- **PoI**：改进的概率

#### 基于信息的功能
- **PES**：预测熵搜索
- **Knowledge_G**：知识梯度

### fit() 方法中的用法

在 `fit()` 方法期间自动选择并应用采集函数。优化过程在内部使用这些函数来推荐下一个实验点。

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

### BGO_Efficient 的高级用法

对于想要显式控制采集功能的高级用户：

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

#### 改进概率 (PoI)

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

#### 增强预期改进 (AEI)

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

#### 预期分位数改进 (EQI)

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

#### 预测熵搜索 (PES)

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

#### 知识梯度（KG）

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



## 基本可视化

Bgolearn 不包含内置可视化方法。但是，您可以使用 matplotlib 轻松创建可视化：

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

## 可视化注意事项

```{note}
Bgolearn 专注于优化算法而不是可视化。对于高级绘图：

1. **使用 matplotlib** 进行基本绘图（如上所示）
2. **使用seaborn**进行统计可视化
3. **使用plotly**进行交互式绘图
4. **使用 BgoFace GUI** 进行视觉优化工作流程

上面的示例展示了如何从 Bgolearn 模型中提取数据并创建自定义可视化。
```
        ----------
        Figsize : 元组，默认=(10, 6)
            图形尺寸（宽度、高度）以英寸为单位
        dpi：整数，默认=100
            图形分辨率
        样式：str，默认='seaborn'
            使用的 Matplotlib 样式
        
        示例
        --------
        >>> 从 bgolearn.visualization 导入 BgolearnVisualizer
        >>> 可视化工具 = BgolearnVisualizer(figsize=(12, 8), dpi=150)
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





## 快速参考

### 常见工作流程

```python
# Basic optimization workflow
from Bgolearn import BGOsampling

# 1. Prepare data
optimizer = BGOsampling.Bgolearn()
model = optimizer.fit(X_train, y_train, X_candidates)

# 2. Single-point optimization
ei_values, next_point = model.EI()

```

### 参数指南

| 函数 | 关键参数 | 推荐值 |
|----------|---------------|-------------------|
| EI | T | None（自动）或自定义阈值 |
| UCB | alpha | 1.0-3.0（越高越偏向探索） |
| PoI | tao | 0.0-0.1（改进容差） |
| AEI | alpha, tao | 1.0-2.0, 0.0-0.2 |
| EQI | beta | 0.25（保守）、0.5（中位）、0.75（乐观） |
| PES | sam_num | 100-500（越高越准确） |
| KG | MC_num | 1-5（越高越准确） |
