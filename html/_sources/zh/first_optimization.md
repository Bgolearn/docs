# 您的第一次优化

```{note}
本页是 Bgolearn 手册的中文版本。
```

```{note}
本教程将逐步引导您完成第一个 Bayesian optimization 和 Bgolearn。
```

## 问题设置

让我们优化一个简单的一维函数来了解工作流程。我们将找到以下最小值：

$$f(x) = (x-2)^2 + 0.1 \sin(10x)$$

该函数在 $x ≈ 1.75$ 附近有一个全局最小值（正弦项使其与 $x = 2$ 略有不同），并且包含振荡使其变得有趣。

## 第 1 步：导入库

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bgolearn import BGOsampling

# Set random seed for reproducibility
np.random.seed(42)
```

## 第 2 步：定义目标函数

```python
def objective_function(x):
    """
    Our test function to minimize: f(x) = (x-2)^2 + 0.1*sin(10x)
    Global minimum at x ≈ 1.77 (sine term shifts it from x=2).
    """
    return (x - 2)**2 + 0.1 * np.sin(10 * x)

# Visualize the function
x_plot = np.linspace(0, 4, 200)
y_plot = [objective_function(x) for x in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Objective function')
plt.axvline(x=1.75, color='red', linestyle='--', alpha=0.7, label='True minimum')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Objective Function to Minimize')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"True minimum: f(1.75) = {objective_function(1.75):.4f}")
```

## 第 3 步：生成初始数据

在真正的优化中，我们从一些初步实验开始：

```python
# Generate initial training data
n_initial = 5
X_train = np.random.uniform(0, 4, n_initial).reshape(-1, 1)
y_train = np.array([objective_function(x[0]) for x in X_train])

# Add some experimental noise
noise_level = 0.02
y_train += noise_level * np.random.randn(len(y_train))

# Convert to pandas (Bgolearn's preferred format)
X_train_df = pd.DataFrame(X_train, columns=['x'])
y_train_series = pd.Series(y_train)

print(f"Initial data: {len(X_train_df)} points")
print(f"Current best: f({X_train_df.iloc[y_train_series.argmin()]['x']:.3f}) = {y_train_series.min():.4f}")
```

## 第 4 步：定义候选点

创建搜索空间 - 我们接下来可能会在其中采样：

```python
# Generate candidate points
X_candidates = np.linspace(0, 4, 100).reshape(-1, 1)
X_candidates_df = pd.DataFrame(X_candidates, columns=['x'])

print(f"Search space: {len(X_candidates_df)} candidate points")
```

## 第 5 步：安装 Bgolearn 模型

```python
# Initialize and fit Bgolearn
optimizer = BGOsampling.Bgolearn()

model = optimizer.fit(
    data_matrix=X_train_df,
    Measured_response=y_train_series,
    virtual_samples=X_candidates_df,
    min_search=True,  # We want to minimize
    CV_test=3,        # 3-fold cross-validation
    Normalize=True    # Normalize features
)

print("Bgolearn model fitted successfully!")
```

## 第 6 步：可视化模型

```python
# Get model predictions
mean_pred = model.virtual_samples_mean
std_pred = model.virtual_samples_std

# Create visualization
plt.figure(figsize=(12, 8))

# Plot true function
plt.plot(x_plot, y_plot, 'b-', linewidth=2, alpha=0.7, label='True function')

# Plot GP mean prediction
plt.plot(X_candidates, mean_pred, 'g-', linewidth=2, label='GP mean')

# Plot uncertainty (±2σ)
plt.fill_between(X_candidates.flatten(), 
                mean_pred - 2*std_pred,
                mean_pred + 2*std_pred,
                alpha=0.3, color='green', label='GP uncertainty (±2σ)')

# Plot training data
plt.scatter(X_train, y_train, c='red', s=100, zorder=5, 
           edgecolors='black', linewidth=1, label='Training data')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gaussian Process Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 第 7 步：使用采集函数

找到下一个最佳采样点：

```python
# Expected Improvement
ei_values, ei_point = model.EI()
print(f"Expected Improvement recommends: x = {ei_point[0][0]:.3f}")

# Upper Confidence Bound
ucb_values, ucb_point = model.UCB(alpha=2.0)
print(f"Upper Confidence Bound recommends: x = {ucb_point[0][0]:.3f}")

# Probability of Improvement
poi_values, poi_point = model.PoI(tao=0.01)
print(f"Probability of Improvement recommends: x = {poi_point[0][0]:.3f}")
```

## 第 8 步：可视化采集函数

```python
# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: GP Model
axes[0,0].plot(x_plot, y_plot, 'b-', linewidth=2, alpha=0.7, label='True function')
axes[0,0].plot(X_candidates, mean_pred, 'g-', linewidth=2, label='GP mean')
axes[0,0].fill_between(X_candidates.flatten(), 
                      mean_pred - 2*std_pred, mean_pred + 2*std_pred,
                      alpha=0.3, color='green', label='GP ±2σ')
axes[0,0].scatter(X_train, y_train, c='red', s=80, zorder=5, 
                 edgecolors='black', label='Training data')
axes[0,0].set_title('Gaussian Process Model')
axes[0,0].set_xlabel('x')
axes[0,0].set_ylabel('f(x)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Expected Improvement
axes[0,1].plot(X_candidates, ei_values, 'purple', linewidth=2)
axes[0,1].axvline(x=ei_point[0][0], color='red', linestyle='--',
                 label=f'EI max (x={ei_point[0][0]:.3f})')
axes[0,1].set_title('Expected Improvement')
axes[0,1].set_xlabel('x')
axes[0,1].set_ylabel('EI(x)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Upper Confidence Bound
axes[1,0].plot(X_candidates, ucb_values, 'orange', linewidth=2)
axes[1,0].axvline(x=ucb_point[0][0], color='red', linestyle='--',
                 label=f'UCB max (x={ucb_point[0][0]:.3f})')
axes[1,0].set_title('Upper Confidence Bound')
axes[1,0].set_xlabel('x')
axes[1,0].set_ylabel('UCB(x)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Probability of Improvement
axes[1,1].plot(X_candidates, poi_values, 'brown', linewidth=2)
axes[1,1].axvline(x=poi_point[0][0], color='red', linestyle='--',
                 label=f'PoI max (x={poi_point[0][0]:.3f})')
axes[1,1].set_title('Probability of Improvement')
axes[1,1].set_xlabel('x')
axes[1,1].set_ylabel('PoI(x)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 第 9 步：模拟下一个实验

```python
# Simulate conducting the experiment recommended by EI
next_x = ei_point[0][0]
true_value = objective_function(next_x)
measured_value = true_value + noise_level * np.random.randn()

print(f"\n🔬 Next Experiment:")
print(f"Recommended point: x = {next_x:.3f}")
print(f"True value: f({next_x:.3f}) = {true_value:.4f}")
print(f"Measured value (with noise): {measured_value:.4f}")
print(f"Current best: {y_train_series.min():.4f}")

if measured_value < y_train_series.min():
    improvement = y_train_series.min() - measured_value
    print(f"Improvement found! Δf = {improvement:.4f}")
else:
    print("No improvement this iteration (normal in optimization)")
```

## 第 10 步：完成优化循环

```python
def run_optimization_loop(n_iterations=10):
    """Run a complete optimization loop."""
    
    # Start with initial data
    X_current = X_train_df.copy()
    y_current = y_train_series.copy()
    
    history = {
        'iteration': [],
        'x_new': [],
        'y_new': [],
        'best_so_far': []
    }
    
    for i in range(n_iterations):
        print(f"\n--- Iteration {i+1} ---")
        
        # Fit model
        model = optimizer.fit(
            data_matrix=X_current,
            Measured_response=y_current,
            virtual_samples=X_candidates_df,
            min_search=True,
            Normalize=True
        )
        
        # Get next point using EI
        ei_values, ei_point = model.EI()
        next_x = ei_point[0][0]
        
        # Evaluate function (simulate experiment)
        true_y = objective_function(next_x)
        measured_y = true_y + noise_level * np.random.randn()
        
        # Update data
        new_row = pd.DataFrame({'x': [next_x]})
        X_current = pd.concat([X_current, new_row], ignore_index=True)
        y_current = pd.concat([y_current, pd.Series([measured_y])], ignore_index=True)
        
        # Track progress
        current_best = y_current.min()
        history['iteration'].append(i+1)
        history['x_new'].append(next_x)
        history['y_new'].append(measured_y)
        history['best_so_far'].append(current_best)
        
        print(f"New point: x = {next_x:.3f}, f(x) = {measured_y:.4f}")
        print(f"Best so far: {current_best:.4f}")
    
    return history, X_current, y_current

# Run optimization
history, X_final, y_final = run_optimization_loop(n_iterations=8)

print(f"\n Final Results:")
print(f"Best found: f({X_final.iloc[y_final.argmin()]['x']:.3f}) = {y_final.min():.4f}")
print(f"True minimum: f(1.750) = {objective_function(1.75):.4f}")
print(f"Error: {abs(y_final.min() - objective_function(1.75)):.4f}")
```

## 第 11 步：分析结果

```python
# Plot optimization progress
plt.figure(figsize=(12, 5))

# Convergence plot
plt.subplot(1, 2, 1)
plt.plot(history['iteration'], history['best_so_far'], 'bo-', linewidth=2, markersize=8)
plt.axhline(y=objective_function(1.75), color='red', linestyle='--',
           label=f'True minimum ({objective_function(1.75):.4f})')
plt.xlabel('Iteration')
plt.ylabel('Best Value Found')
plt.title('Optimization Convergence')
plt.legend()
plt.grid(True, alpha=0.3)

# Sampling locations
plt.subplot(1, 2, 2)
plt.plot(x_plot, y_plot, 'b-', linewidth=2, alpha=0.7, label='True function')
plt.scatter(X_train, y_train, c='red', s=100, marker='o', 
           edgecolors='black', label='Initial points', zorder=5)
plt.scatter(history['x_new'], history['y_new'], c='green', s=100, marker='s', 
           edgecolors='black', label='BO points', zorder=5)
plt.axvline(x=1.75, color='red', linestyle='--', alpha=0.7, label='True minimum')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Sampling Locations')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 要点

**我们学到了什么：**

1. **Bayesian optimization 是迭代的** - 每个新点都会改进我们的模型
2. **获取功能平衡探索与利用** - EI 找到了良好的平衡
3. **不确定性指导抽样** - 探索具有高不确定性的点
4. **需要很少的评估** - 我们仅通过几次实验就找到了最小值

**后续步骤：**

- 尝试不同的采集函数：{doc}`acquisition_functions`
- 探索材料应用：{doc}`materials_discovery`
- 了解多目标优化：{doc}`multibgolearn`
- 尝试GUI界面：{doc}`bgoface`

```{tip}
这个简单的示例演示了核心 Bgolearn 工作流程。同样的原则适用于具有多个目标和约束的复杂材料发现问题！
```
