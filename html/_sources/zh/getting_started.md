# 开始使用 Bgolearn

```{note}
本页是 Bgolearn 手册的中文版本。
```

```{note}
本指南将帮助您安装 Bgolearn 并在短短几分钟内运行您的第一次优化。
```

## 安装

### 先决条件

Bgolearn 需要 Python 3.7 或更高版本。我们建议使用虚拟环境：

```bash
# Create virtual environment
python -m venv bgolearn_env

# Activate virtual environment
# On Windows:
bgolearn_env\Scripts\activate
# On macOS/Linux:
source bgolearn_env/bin/activate
```

### 安装Bgolearn

从 PyPI 安装主包：

```bash
pip install Bgolearn
```

对于多目标优化，还要安装 MultiBgolearn：

```bash
pip install MultiBgolearn
```

或者将两者安装在一起：

```bash
pip install Bgolearn MultiBgolearn
```

### 验证安装

测试您的安装：

```python
# Test single-objective Bgolearn
from Bgolearn import BGOsampling
print("Bgolearn imported successfully!")

# Test multi-objective MultiBgolearn
try:
    from MultiBgolearn import bgo
    print("MultiBgolearn imported successfully!")
except ImportError:
    print("MultiBgolearn not installed. Install with: pip install MultiBgolearn")
```

## 基本概念

### Bayesian Optimization是什么？

Bayesian optimization 是一种用于优化评估成本高昂的函数的强大技术。它在以下情况下特别有用：
- 实验是昂贵的（时间、金钱、资源）
- 函数评估有噪音
- 渐变不可用
- 您想尽量减少实验次数

### 关键部件

1. **代理模型**：近似未知函数（通常为 Gaussian Process 等）
2. **采集功能**：决定下一步采样的位置
3. **优化循环**：迭代改进代理模型

## 您的第一次优化

### 第 1 步：生成样本数据

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Generate synthetic materials data
def create_materials_dataset(n_samples=50, n_features=4, noise=0.1):
    """Create synthetic materials property data."""
    np.random.seed(42)
    
    # Generate base features
    X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                          noise=noise, random_state=42)
    
    # Create realistic feature names
    feature_names = ['Temperature', 'Pressure', 'Composition_A', 'Composition_B']
    
    # Normalize features to realistic ranges
    X[:, 0] = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min()) * 500 + 300  # Temperature: 300-800K
    X[:, 1] = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min()) * 10 + 1     # Pressure: 1-11 GPa
    X[:, 2] = np.abs(X[:, 2]) / np.abs(X[:, 2]).max() * 0.8 + 0.1  # Composition: 0.1-0.9
    X[:, 3] = 1 - X[:, 2]  # Ensure compositions sum to 1
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Strength'] = y  # Target property (e.g., material strength)
    
    return df

# Create training data
train_data = create_materials_dataset(n_samples=30)
print("Training data shape:", train_data.shape)
print("\nFirst 5 rows:")
print(train_data.head())
```

### 第 2 步：准备优化数据

```python
from Bgolearn import BGOsampling

# Separate features and target
X_train = train_data.drop('Strength', axis=1)
y_train = train_data['Strength']

# Create virtual candidates for optimization
def create_candidate_materials(n_candidates=200):
    """Create candidate materials for optimization."""
    np.random.seed(123)
    
    # Generate candidate space
    candidates = []
    for _ in range(n_candidates):
        temp = np.random.uniform(300, 800)  # Temperature
        pressure = np.random.uniform(1, 11)  # Pressure
        comp_a = np.random.uniform(0.1, 0.9)  # Composition A
        comp_b = 1 - comp_a  # Composition B
        
        candidates.append([temp, pressure, comp_a, comp_b])
    
    return pd.DataFrame(candidates, columns=X_train.columns)

X_candidates = create_candidate_materials()
print(f"Created {len(X_candidates)} candidate materials")
```

### 步骤3：初始化并拟合Bgolearn

```python
# Initialize Bgolearn optimizer
optimizer = BGOsampling.Bgolearn()

# Fit the model
print("Fitting Bgolearn model...")
model = optimizer.fit(
    data_matrix=X_train,
    Measured_response=y_train,
    virtual_samples=X_candidates,
    Mission='Regression',
    min_search=False,  # We want to maximize strength
    CV_test=5,  # 5-fold cross-validation
    Normalize=True
)

print("Model fitted successfully!")
```

### 第四步：单点优化

```python
# Expected Improvement
print("\n=== Expected Improvement ===")
ei_values, next_point_ei = model.EI()
print(f"Next experiment (EI): {next_point_ei}")

# Upper Confidence Bound
print("\n=== Upper Confidence Bound ===")
ucb_values, next_point_ucb = model.UCB(alpha=2.0)
print(f"Next experiment (UCB): {next_point_ucb}")

# Probability of Improvement
print("\n=== Probability of Improvement ===")
poi_values, next_point_poi = model.PoI(tao=0.01)
print(f"Next experiment (PoI): {next_point_poi}")
```


### 第 5 步：基本可视化

```python
import matplotlib.pyplot as plt

# Get EI values for visualization
ei_values, recommended_points = model.EI()

# Plot Expected Improvement values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(ei_values)
plt.title("Expected Improvement Values")
plt.xlabel("Candidate Index")
plt.ylabel("EI Value")
plt.grid(True)

# Plot predicted vs actual (if you have true function)
plt.subplot(1, 2, 2)
plt.scatter(range(len(model.virtual_samples_mean)), model.virtual_samples_mean, alpha=0.6)
plt.title("Predicted Values for All Candidates")
plt.xlabel("Candidate Index")
plt.ylabel("Predicted Value")
plt.grid(True)

plt.tight_layout()
plt.show()

```

## 了解结果

### 解释采集函数

1. **Expected Improvement (EI)**：
   - 平衡勘探和开发
   - 值越高表示区域更有前景
   - 良好的通用选择

2. **Upper Confidence Bound (UCB)**：
   - 由参数`alpha`控制
   - 更高的 `alpha` = 更多探索
   - 适合嘈杂的功能

3. **Probability of Improvement（PoI）**：
   - 简单直观
   - 由参数`tao`控制
   - 可能过度剥削

### 模型验证

```python
# Check cross-validation results
print("\n=== Model Validation ===")
print("Cross-validation results saved in ./Bgolearn/ directory")

# List generated files
import os
bgo_files = [f for f in os.listdir('./Bgolearn') if f.endswith('.csv')]
print("Generated files:")
for file in bgo_files:
    print(f"  - {file}")
```

## 下一步

现在您已经完成了第一次优化，请探索以下高级主题：

1. **[Acquisition Functions](acquisition_functions.md)** - 深入探讨不同的收购策略
2. **[Batch Optimization](batch_optimization.md)** - 并行实验设计
3. **[Visualization](visualization.md)** - 高级绘图和仪表板
4. **[Materials Discovery](materials_discovery.md)** - 专业工作流程

## 成功秘诀

### 数据准备
- 确保要素具有相似的比例（使用 `Normalize=True`）
- 删除高度相关的特征
- 适当处理缺失值

### 选型
- 默认Gaussian Process开始
- 使用交叉验证来评估模型质量
- 考虑实验中的噪音水平

### 采集功能选择
- EI：良好的一般选择
- UCB：更适合嘈杂的函数
- 批处理方法：当您可以运行并行实验时

### 迭代策略
- 从空间填充设计开始
- 使用采集函数进行细化
- 仔细监控收敛

## 常见问题

### 内存问题
```python
# For large candidate sets, use batching
if len(X_candidates) > 100000:
    print("Large candidate set detected. Consider using smaller batches.")
```

### 收敛问题
```python
# Check for proper normalization
print("Feature ranges:")
print(X_train.describe())

# Ensure sufficient training data
if len(X_train) < 10:
    print("Warning: Very small training set. Consider collecting more data.")
```

### 数值稳定性
```python
# Check for extreme values
print("Target variable statistics:")
print(y_train.describe())

# Look for outliers
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
outliers = y_train[(y_train < Q1 - 1.5*IQR) | (y_train > Q3 + 1.5*IQR)]
print(f"Potential outliers: {len(outliers)}")
```
