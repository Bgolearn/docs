# 采集功能指南

```{note}
本页是 Bgolearn 手册的中文版本。
```

## 介绍

获取函数是贝叶斯优化的核心。他们通过平衡**探索**（在不确定区域中采样）和**利用**（在当前最佳点附近采样）来确定下一步在哪里采样。

## 可用的采集功能

### 预期改善 (EI)

最流行的获取功能，自然地平衡探索和利用。

**数学公式：**
```
EI(x) = (μ(x) - f*) * Φ(Z) + σ(x) * φ(Z)
where Z = (μ(x) - f*) / σ(x)
```

**用法：**
```python
from Bgolearn import BGOsampling
import numpy as np
import pandas as pd

# Create sample data
def create_test_function():
    """Create a test optimization problem."""
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (20, 2))
    y = -(X[:, 0]**2 + X[:, 1]**2) + 0.1 * np.random.randn(20)  # Negative quadratic with noise
    return pd.DataFrame(X, columns=['x1', 'x2']), pd.Series(y)

# Generate candidates
def create_candidates():
    x1 = np.linspace(-3, 3, 50)
    x2 = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1, x2)
    candidates = np.column_stack([X1.ravel(), X2.ravel()])
    return pd.DataFrame(candidates, columns=['x1', 'x2'])

X_train, y_train = create_test_function()
X_candidates = create_candidates()

# Initialize and fit
optimizer = BGOsampling.Bgolearn()
model = optimizer.fit(X_train, y_train, X_candidates, min_search=False)

# Expected Improvement
ei_values, next_point = model.EI()
print(f"EI selected point: {next_point}")
print(f"EI values range: [{ei_values.min():.4f}, {ei_values.max():.4f}]")

# Expected Improvement with custom baseline
ei_values_custom, next_point_custom = model.EI(T=-2.0)  # Custom threshold
print(f"EI with custom baseline: {next_point_custom}")
```

**何时使用：**
- 通用优化
- 平衡探索-利用
- 大多数研究应用


### 对数预期改进 

对 EI 值应用对数变换以放大细微差异。

```python
# logarithmic Expected Improvement 
eilog_values, next_point_eilog = model.EI_log()
print(f"Logarithmic EI selected point: {next_point_eilog}")

# Compare with standard EI
print(f"Standard EI point: {next_point}")
print(f"Logarithmic EI point: {next_point_eilog}")
```






### 插件的预期改进

使用模型对训练数据的预测作为基线，而不是实际的最佳观测值。

```python
# Expected Improvement with Plugin
eip_values, next_point_eip = model.EI_plugin()
print(f"EI Plugin selected point: {next_point_eip}")

# Compare with standard EI
print(f"Standard EI point: {next_point}")
print(f"Plugin EI point: {next_point_eip}")
```

**何时使用：**
- 当训练数据有噪声时
- 模型预测比观测更可靠

### 置信上限 (UCB)

乐观方法选择具有高预测平均值和不确定性的点。

**数学公式：**
```
UCB(x) = μ(x) + α * σ(x)
```

```python
# Upper Confidence Bound with different exploration parameters
ucb_conservative, point_conservative = model.UCB(alpha=1.0)  # Conservative
ucb_balanced, point_balanced = model.UCB(alpha=2.0)  # Balanced (default)
ucb_exploratory, point_exploratory = model.UCB(alpha=3.0)  # Exploratory

print("UCB Results:")
print(f"Conservative (α=1.0): {point_conservative}")
print(f"Balanced (α=2.0): {point_balanced}")
print(f"Exploratory (α=3.0): {point_exploratory}")

# Visualize the effect of alpha
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
alphas = [1.0, 2.0, 3.0]
titles = ['Conservative', 'Balanced', 'Exploratory']

for i, (alpha, title) in enumerate(zip(alphas, titles)):
    ucb_vals, _ = model.UCB(alpha=alpha)
    
    # Reshape for 2D plot (assuming square grid)
    grid_size = int(np.sqrt(len(ucb_vals)))
    ucb_grid = ucb_vals[:grid_size**2].reshape(grid_size, grid_size)
    
    im = axes[i].imshow(ucb_grid, extent=[-3, 3, -3, 3], origin='lower', cmap='viridis')
    axes[i].set_title(f'{title} (α={alpha})')
    axes[i].set_xlabel('x1')
    axes[i].set_ylabel('x2')
    plt.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.show()
```

**何时使用：**
- 嘈杂的函数
- 当你想明确控制探索时
- 多臂老虎机问题

### 改进概率 (PoI)

选择较当前最佳状态改进概率最高的点。

**数学公式：**
```
PoI(x) = Φ((μ(x) - f* - τ) / σ(x))
```

```python
# Probability of Improvement with different thresholds
poi_strict, point_strict = model.PoI(tao=0.0)  # Strict improvement
poi_relaxed, point_relaxed = model.PoI(tao=0.1)  # Allow small degradation
poi_very_relaxed, point_very_relaxed = model.PoI(tao=0.5)  # Very relaxed

print("PoI Results:")
print(f"Strict (τ=0.0): {point_strict}")
print(f"Relaxed (τ=0.1): {point_relaxed}")
print(f"Very Relaxed (τ=0.5): {point_very_relaxed}")

# Custom baseline
poi_custom, point_custom = model.PoI(tao=0.1, T=-1.5)
print(f"Custom baseline (T=-1.5): {point_custom}")
```

**何时使用：**
- 当任何改进都有价值时
- 风险规避优化
- 概念验证研究

### 增强预期改进 (AEI)

EI 的增强版本，可考虑噪声并使用更复杂的基线。

```python
# Augmented Expected Improvement
aei_default, point_aei_default = model.Augmented_EI()
aei_custom, point_aei_custom = model.Augmented_EI(alpha=1.5, tao=0.1)

print("AEI Results:")
print(f"Default parameters: {point_aei_default}")
print(f"Custom (α=1.5, τ=0.1): {point_aei_custom}")
```

**参数：**
- `alpha`：基线选择的权衡系数
- `tao`：噪声标准差

**何时使用：**
- 实验数据有噪声
- 当您想要更复杂的基线选择时

### 预期分位数改进 (EQI)

使用分位数而不是平均值来实现更稳健的优化。

```python
# Expected Quantile Improvement
eqi_median, point_eqi_median = model.EQI(beta=0.5)  # Median
eqi_conservative, point_eqi_conservative = model.EQI(beta=0.25)  # 25th percentile
eqi_optimistic, point_eqi_optimistic = model.EQI(beta=0.75)  # 75th percentile

print("EQI Results:")
print(f"Median (β=0.5): {point_eqi_median}")
print(f"Conservative (β=0.25): {point_eqi_conservative}")
print(f"Optimistic (β=0.75): {point_eqi_optimistic}")
```

**何时使用：**
- 稳健优化
- 当您想要优化特定分位数时
- 风险管理场景

### 重新插值预期改进 (REI)

使用重新插值进行基线计算。

```python
# Reinterpolation Expected Improvement
rei_values, point_rei = model.Reinterpolation_EI()
print(f"REI selected point: {point_rei}")
```

**何时使用：**
- 当训练数据的模型预测不可靠时
- 迭代细化场景

### 预测熵搜索 (PES)

最大化有关最优值的信息增益的信息论方法。

```python
# Predictive Entropy Search
pes_default, point_pes_default = model.PES()  # Default 100 samples
pes_high_precision, point_pes_high = model.PES(sam_num=500)  # Higher precision

print("PES Results:")
print(f"Default (100 samples): {point_pes_default}")
print(f"High precision (500 samples): {point_pes_high}")
```

**何时使用：**
- 信息论优化
- 当您想最大限度地了解最佳方案时
- 学术研究

### 知识梯度（KG）

估计每个潜在测量的信息价值。

```python
# Knowledge Gradient
kg_fast, point_kg_fast = model.Knowledge_G(MC_num=1)  # Fast approximation
kg_accurate, point_kg_accurate = model.Knowledge_G(MC_num=5)  # More accurate

print("KG Results:")
print(f"Fast (1 MC sample): {point_kg_fast}")
print(f"Accurate (5 MC samples): {point_kg_accurate}")

# Parallel version (if supported)
try:
    kg_parallel, point_kg_parallel = model.Knowledge_G(MC_num=3, Proc_num=2)
    print(f"Parallel (2 processes): {point_kg_parallel}")
except:
    print("Parallel processing not available on this system")
```

**何时使用：**
- 当测量成本变化时
- 投资组合优化
- 多保真度优化
<!-- 
### 汤普森采样

从后验分布中采样的概率方法。

```python
# Thompson Sampling
ts_point, ts_value, ts_uncertainty = model.Thompson_sampling()

print("Thompson Sampling Results:")
print(f"Selected point: {ts_point}")
print(f"Sampled value: {ts_value:.4f}")
print(f"Uncertainty: {ts_uncertainty:.4f}")

# Multiple Thompson samples
print("\nMultiple Thompson samples:")
for i in range(5):
    point, value, uncertainty = model.Thompson_sampling()
    print(f"Sample {i+1}: point={point}, value={value:.4f}")
``` -->

**When to use:**
- Exploration-heavy scenarios
- Multi-armed bandit problems
- When you want diverse sampling strategies

## Sequential Optimization Strategy

For multiple experiments, use sequential optimization by updating the model:

```python
# Sequential optimization example
def sequential_optimization(model, n_iterations=3):
    """Perform sequential optimization."""
    results = []

    for i in range(n_iterations):
        print(f"\n=== Iteration {i+1} ===")

        # Get recommendation using EI
        ei_values, recommended_points = model.EI()
        next_point = recommended_points[0]

        print(f"Recommended point: {next_point}")

        # Simulate experiment (replace with actual experiment)
        # For demo, we'll use the true function
        simulated_result = -(next_point[0]**2 + next_point[1]**2) + 0.1 * np.random.randn()

        results.append({
            'iteration': i+1,
            'point': next_point,
            'result': simulated_result
        })

        print(f"Experimental result: {simulated_result:.4f}")

        # In practice, you would retrain the model with new data here
        # model = retrain_with_new_data(model, next_point, simulated_result)

    return results

# Run sequential optimization
optimization_results = sequential_optimization(model, n_iterations=3)
```

## 表 2.1：Bgolearn 中完整的采集功能

| No. | Function Name | Method Call | Parameters | Return Format | Mathematical Basis | Primary Use Case |
|-----|---------------|-------------|------------|---------------|-------------------|------------------|
| 1 | **Expected Improvement** | `model.EI(T=None)` | `T`: baseline threshold (float, optional) | `(values, points)` | EI(x) = (μ(x) - f*) × Φ(Z) + σ(x) × φ(Z) | General-purpose optimization |
| 2 | **Logarithmic Expected Improvement** | `model.EI_log(T=None)` | `T`: baseline threshold (float, optional) | `(values, points)` | EI_log(x) = log[(μ(x) - f*) × Φ(Z) + σ(x) × φ(Z)]| General-purpose optimization |
| 3 | **EI with Plugin** | `model.EI_plugin()` | None | `(values, points)` | EI with model prediction baseline | Noisy experimental data |
| 4 | **Augmented Expected Improvement** | `model.Augmented_EI(alpha=1, tao=0)` | `alpha`: trade-off coefficient (float)<br>`tao`: noise std (float) | `(values, points)` | Enhanced EI with noise consideration | Noisy optimization problems |
| 5 | **Expected Quantile Improvement** | `model.EQI(beta=0.5, tao_new=0)` | `beta`: quantile level (0-1)<br>`tao_new`: noise std (float) | `(values, points)` | Quantile-based improvement | Robust optimization |
| 6 | **Reinterpolation EI** | `model.Reinterpolation_EI()` | None | `(values, points)` | EI with reinterpolated baseline | Model refinement scenarios |
| 7 | **Upper Confidence Bound** | `model.UCB(alpha=1)` | `alpha`: exploration weight (float) | `(values, points)` | UCB(x) = μ(x) ± α × σ(x) | High exploration needs |
| 8 | **Probability of Improvement** | `model.PoI(tao=0, T=None)` | `tao`: improvement threshold (float)<br>`T`: baseline (float, optional) | `(values, points)` | PoI(x) = Φ((μ(x) - f* - τ) / σ(x)) | Risk-averse optimization |
| 9 | **Predictive Entropy Search** | `model.PES(sam_num=500)` | `sam_num`: Monte Carlo samples (int) | `(values, points)` | Information-theoretic approach | Maximum information gain |
| 10 | **Knowledge Gradient** | `model.Knowledge_G(MC_num=1, Proc_num=None)` | `MC_num`: MC simulations (1-10)<br>`Proc_num`: processors (int, optional) | `(values, points)` | Value of information estimation | Multi-fidelity optimization |
| 11 | **Thompson Sampling** | `model.Thompson_sampling()` | None | `(point, value, uncertainty)` | Posterior sampling | Diverse exploration strategies |

### 功能类别

#### **基于改进的功能**
- **EI、EI_log、EI_plugin、Augmented_EI、EQI、Reinterpolation_EI**：关注相对于当前最佳值的预期改进
- **数学基础**：基于改进概率和幅度
- **最适合**：一般优化、噪声数据、鲁棒优化

#### **基于置信度的函数**
- **UCB、PoI**：基于置信界限和改进概率
- **数学基础**：统计置信区间
- **最适合**：勘探控制、风险管理

#### **信息论函数**
- **PES、Knowledge_G**：最大化有关最优值的信息增益
- **数学基础**：熵和信息论
- **最适合**：了解函数、多保真度问题

#### **基于采样的函数**
- **Thompson_sampling**：后验概率采样
- **数学基础**：贝叶斯后验采样
- **最适合**：多样化探索，多臂强盗

### 表 2.2：参数详细信息和默认值

| Function | Parameter | Type | Default | Range/Options | Description | Effect |
|----------|-----------|------|---------|---------------|-------------|--------|
| **EI** | `T` | float | `None` | Any float or `None` | Baseline threshold for improvement | Lower T → more exploration |
| **EI_log** | `T` | float | `None` | Any float or `None` | Baseline threshold for improvement | Lower T → more exploration |
| **Augmented_EI** | `alpha` | float | `1` | > 0 | Trade-off coefficient | Higher α → more conservative baseline |
| | `tao` | float | `0` | ≥ 0 | Noise standard deviation | Higher τ → accounts for more noise |
| **EQI** | `beta` | float | `0.5` | [0, 1] | Quantile level (0.5 = median) | Lower β → more conservative |
| | `tao_new` | float | `0` | ≥ 0 | Noise standard deviation | Higher τ → more robust |
| **UCB** | `alpha` | float | `1` | > 0 | Exploration weight | Higher α → more exploration |
| **PoI** | `tao` | float | `0` | ≥ 0 | Improvement threshold | Higher τ → easier to satisfy |
| | `T` | float | `None` | Any float or `None` | Baseline threshold | Custom baseline override |
| **PES** | `sam_num` | int | `500` | > 0 | Monte Carlo samples | Higher → more accurate, slower |
| **Knowledge_G** | `MC_num` | int | `1` | 1-10 | Monte Carlo simulations | Higher → more accurate, slower |
| | `Proc_num` | int | `None` | > 0 or `None` | Number of processors | Parallel processing control |

### 表 2.3：勘探与开发特征

| Function | Exploration Level | Exploitation Level | Balance Type | Tuning Parameter |
|----------|------------------|-------------------|--------------|------------------|
| **EI** | Medium | Medium | **Balanced** | `T` (baseline) |
| **EI_log** | Medium | Medium | **Balanced** | `T` (baseline) |
| **EI_plugin** | Medium | Medium | **Balanced** | None |
| **Augmented_EI** | Medium-High | Medium | **Balanced+** | `alpha`, `tao` |
| **EQI** | Medium | Medium-High | **Balanced** | `beta` |
| **Reinterpolation_EI** | Medium | Medium | **Balanced** | None |
| **UCB** | **High** | Low-Medium | **Exploration-heavy** | `alpha` |
| **PoI** | Low | **High** | **Exploitation-heavy** | `tao` |
| **PES** | **High** | Low | **Exploration-heavy** | `sam_num` |
| **Knowledge_G** | Medium-High | Medium | **Balanced** | `MC_num` |
| **Thompson_sampling** | **High** | Low | **Exploration-heavy** | None |

### 快速选型指南

**对于初学者**：从**预期改进 (EI)** 开始 - 它是使用最广泛且易于理解的。

**对于噪声数据**：使用具有适当参数的 **增强 EI** 或 **UCB**。

**对于规避风险的优化**：使用**改进概率 (PoI)**。

**为了最大程度地探索**：使用具有高 alpha 的 **UCB** 或 **Thompson 采样**。

**对于信息论方法**：使用 **PES** 或 **知识梯度**。

**为了稳健优化**：使用**预期分位数改进 (EQI)**。

## 表 2.4：完整的使用示例

```python
# Complete demonstration of all acquisition functions
from Bgolearn import BGOsampling
import numpy as np
import pandas as pd

# Create sample optimization problem
def demonstrate_all_acquisition_functions():
    """Demonstrate all 10 acquisition functions in Bgolearn."""

    # Setup data
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(15, 2), columns=['x1', 'x2'])
    y_train = pd.Series(-(X_train['x1']**2 + X_train['x2']**2) + 0.1*np.random.randn(15))
    X_candidates = pd.DataFrame(np.random.randn(50, 2), columns=['x1', 'x2'])

    # Fit model
    optimizer = BGOsampling.Bgolearn()
    model = optimizer.fit(X_train, y_train, X_candidates, opt_num=1)

    print("All Bgolearn Acquisition Functions Demo")
    print("=" * 60)

    # 1. Expected Improvement
    print("\n1. Expected Improvement (EI)")
    ei_values, ei_point = model.EI()
    print(f"   Selected point: {ei_point[0]}")
    print(f"   EI range: [{ei_values.min():.4f}, {ei_values.max():.4f}]")

    # 2. EI with Plugin
    print("\n2. Expected Improvement with Plugin")
    eip_values, eip_point = model.EI_plugin()
    print(f"   Selected point: {eip_point[0]}")
    print(f"   EI Plugin range: [{eip_values.min():.4f}, {eip_values.max():.4f}]")

    # 3. Augmented EI
    print("\n3. Augmented Expected Improvement")
    aei_values, aei_point = model.Augmented_EI(alpha=1.5, tao=0.1)
    print(f"   Selected point: {aei_point[0]}")
    print(f"   AEI range: [{aei_values.min():.4f}, {aei_values.max():.4f}]")

    # 4. Expected Quantile Improvement
    print("\n4. Expected Quantile Improvement")
    eqi_values, eqi_point = model.EQI(beta=0.5, tao_new=0.05)
    print(f"   Selected point: {eqi_point[0]}")
    print(f"   EQI range: [{eqi_values.min():.4f}, {eqi_values.max():.4f}]")

    # 5. Reinterpolation EI
    print("\n5. Reinterpolation Expected Improvement")
    rei_values, rei_point = model.Reinterpolation_EI()
    print(f"   Selected point: {rei_point[0]}")
    print(f"   REI range: [{rei_values.min():.4f}, {rei_values.max():.4f}]")

    # 6. Upper Confidence Bound
    print("\n6. Upper Confidence Bound")
    ucb_values, ucb_point = model.UCB(alpha=2.0)
    print(f"   Selected point: {ucb_point[0]}")
    print(f"   UCB range: [{ucb_values.min():.4f}, {ucb_values.max():.4f}]")

    # 7. Probability of Improvement
    print("\n7. Probability of Improvement")
    poi_values, poi_point = model.PoI(tao=0.01)
    print(f"   Selected point: {poi_point[0]}")
    print(f"   PoI range: [{poi_values.min():.4f}, {poi_values.max():.4f}]")

    # 8. Predictive Entropy Search
    print("\n8. Predictive Entropy Search")
    pes_values, pes_point = model.PES(sam_num=100)
    print(f"   Selected point: {pes_point[0]}")
    print(f"   PES range: [{pes_values.min():.4f}, {pes_values.max():.4f}]")

    # 9. Knowledge Gradient
    print("\n9. Knowledge Gradient")
    kg_values, kg_point = model.Knowledge_G(MC_num=1)
    print(f"   Selected point: {kg_point[0]}")
    print(f"   KG range: [{kg_values.min():.4f}, {kg_values.max():.4f}]")

    # 10. Thompson Sampling (different return format)
    print("\n10. Thompson Sampling")
    ts_point, ts_value, ts_uncertainty = model.Thompson_sampling()
    print(f"   Selected point: {ts_point}")
    print(f"   Sampled value: {ts_value:.4f}")
    print(f"   Uncertainty: {ts_uncertainty:.4f}")

    print("\n" + "=" * 60)
    print("All 10 acquisition functions demonstrated successfully!")

# Run the demonstration
demonstrate_all_acquisition_functions()
```

## 比较采集功能

```python
# Compare all acquisition functions
def compare_acquisition_functions(model):
    """Compare different acquisition functions."""
    results = {}

    # Standard acquisition functions (return values, points)
    methods = {
        'EI': lambda: model.EI(),
        'EI_Plugin': lambda: model.EI_plugin(),
        'Augmented_EI': lambda: model.Augmented_EI(),
        'EQI': lambda: model.EQI(beta=0.5),
        'Reinterpolation_EI': lambda: model.Reinterpolation_EI(),
        'UCB': lambda: model.UCB(alpha=2.0),
        'PoI': lambda: model.PoI(tao=0.01),
        'PES': lambda: model.PES(sam_num=100),
        'Knowledge_G': lambda: model.Knowledge_G(MC_num=1)
    }

    # Thompson Sampling (different return format)
    ts_methods = {
        'Thompson_Sampling': lambda: model.Thompson_sampling()
    }
    
    for name, method in methods.items():
        try:
            values, point = method()
            results[name] = {
                'point': point[0] if len(point.shape) > 1 else point,
                'max_value': values.max(),
                'mean_value': values.mean(),
                'std_value': values.std()
            }
        except Exception as e:
            print(f"Error with {name}: {e}")
            results[name] = None

    # Handle Thompson Sampling separately (different return format)
    for name, method in ts_methods.items():
        try:
            point, value, uncertainty = method()
            results[name] = {
                'point': point,
                'sampled_value': value,
                'uncertainty': uncertainty
            }
        except Exception as e:
            print(f"Error with {name}: {e}")
            results[name] = None

    return results

# Run comparison
comparison = compare_acquisition_functions(model)

# Display results
print("Acquisition Function Comparison:")
print("-" * 70)
for name, result in comparison.items():
    if result:
        print(f"{name:15s}: Point={result['point']}")
        if 'max_value' in result:
            print(f"                Max={result['max_value']:.4f}, "
                  f"Mean={result['mean_value']:.4f}, "
                  f"Std={result['std_value']:.4f}")
        elif 'sampled_value' in result:
            print(f"                Sampled Value={result['sampled_value']:.4f}, "
                  f"Uncertainty={result['uncertainty']:.4f}")
    else:
        print(f"{name:15s}: Failed")
    print()
```

## 可视化采集功能

```python
import matplotlib.pyplot as plt
import numpy as np

# Basic visualization using matplotlib

# Example: 1D acquisition function visualization
def visualize_1d_acquisition():
    """Visualize acquisition functions for 1D problems."""
    from Bgolearn import BGOsampling
    import pandas as pd

    # Create 1D test problem
    x_1d = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_1d = -(x_1d.flatten() - 1)**2 + 0.1 * np.random.randn(100)

    # Select training points
    train_indices = np.random.choice(100, 10, replace=False)
    X_train_1d = x_1d[train_indices]
    y_train_1d = y_1d[train_indices]

    # Fit model
    optimizer_1d = BGOsampling.Bgolearn()
    model_1d = optimizer_1d.fit(
        pd.DataFrame(X_train_1d, columns=['x']),
        pd.Series(y_train_1d),
        pd.DataFrame(x_1d, columns=['x']),
        min_search=False
    )

    # Get acquisition values
    ei_vals, ei_point = model_1d.EI()

    # Create custom plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Objective function and model
    x_plot = x_1d.flatten()
    ax1.plot(x_plot, y_1d, 'b-', alpha=0.3, label='True function')
    ax1.scatter(X_train_1d.flatten(), y_train_1d, c='red', s=50, label='Training data')
    ax1.plot(x_plot, model_1d.virtual_samples_mean, 'g-', label='GP mean')
    ax1.fill_between(x_plot,
                     model_1d.virtual_samples_mean - model_1d.virtual_samples_std,
                     model_1d.virtual_samples_mean + model_1d.virtual_samples_std,
                     alpha=0.2, color='green', label='GP uncertainty')
    ax1.axvline(x=ei_point[0], color='orange', linestyle='--', label='Next point')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Gaussian Process Model')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Acquisition function
    ax2.plot(x_plot, ei_vals, 'purple', linewidth=2, label='Expected Improvement')
    ax2.axvline(x=ei_point[0], color='orange', linestyle='--', label='Next point')
    ax2.set_xlabel('x')
    ax2.set_ylabel('EI Value')
    ax2.set_title('Expected Improvement Acquisition Function')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Run 1D visualization
visualize_1d_acquisition()
```

## 选择正确的采集功能

### 决策树

```
Are you doing parallel experiments?
├─ Yes → Use batch_EI() or batch_UCB()
└─ No → Continue below

Is your function very noisy?
├─ Yes → Use UCB (high α) or AEI
└─ No → Continue below

Do you want balanced exploration/exploitation?
├─ Yes → Use EI (recommended for most cases)
└─ No → Continue below

Do you want more exploration?
├─ Yes → Use UCB (high α) or PES
└─ No → Use PoI or EQI
```

### 性能特点

| Function | Exploration | Exploitation | Noise Robustness | Computational Cost |
|----------|-------------|--------------|------------------|-------------------|
| EI       | Medium      | Medium       | Medium           | Low               |
| UCB      | High        | Low          | High             | Low               |
| PoI      | Low         | High         | Low              | Low               |
| AEI      | Medium      | Medium       | High             | Medium            |
| EQI      | Medium      | Medium       | High             | Medium            |
| REI      | Medium      | Medium       | Medium           | Medium            |
| PES      | High        | Low          | Medium           | High              |
| KG       | Medium      | Medium       | Medium           | High              |

## 高级提示

### 自定义采集功能

```python
# You can implement custom acquisition functions
class CustomAcquisition:
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.current_best = y_train.max()  # For maximization
    
    def custom_ei_variant(self, X_candidates, beta=1.0):
        """Custom EI variant with additional parameter."""
        mean, std = self.model.fit_pre(self.X_train, self.y_train, X_candidates)
        
        improvement = mean - self.current_best
        z = improvement / (std + 1e-9)
        
        # Custom modification: add beta parameter
        ei = improvement * norm.cdf(z) + beta * std * norm.pdf(z)
        return np.maximum(ei, 0)

# Usage would require integration with the main framework
```

### 参数调整

```python
# Systematic parameter exploration
def tune_acquisition_parameters():
    """Tune acquisition function parameters."""
    
    # UCB alpha tuning
    alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ucb_results = {}
    
    for alpha in alphas:
        ucb_vals, point = model.UCB(alpha=alpha)
        ucb_results[alpha] = {
            'point': point,
            'max_acquisition': ucb_vals.max()
        }
    
    print("UCB Alpha Tuning:")
    for alpha, result in ucb_results.items():
        print(f"α={alpha}: max_acq={result['max_acquisition']:.4f}")
    
    # PoI tau tuning
    taus = [0.0, 0.01, 0.05, 0.1, 0.2]
    poi_results = {}
    
    for tau in taus:
        poi_vals, point = model.PoI(tao=tau)
        poi_results[tau] = {
            'point': point,
            'max_acquisition': poi_vals.max()
        }
    
    print("\nPoI Tau Tuning:")
    for tau, result in poi_results.items():
        print(f"τ={tau}: max_acq={result['max_acquisition']:.4f}")

tune_acquisition_parameters()
```
