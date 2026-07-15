# Bgolearn 中的代理模型

```{note}
本页是 Bgolearn 手册的中文版本。
```

```{warning}
**重要**：所有数据必须是 pandas DataFrames/Series，而不是 numpy 数组！

- `data_matrix` → `pd.DataFrame` 带列名
- `Measured_response` → `pd.Series`
- `virtual_samples` → `pd.DataFrame` 具有相同的列名

使用numpy数组会导致：`AttributeError: 'numpy.ndarray' object has no attribute 'columns'`
```

```{note}
本页介绍了 Bgolearn 中可用的不同代理模型以及如何为您的优化问题选择正确的模型。
```

## 概述

代理模型（也称为元模型）是贝叶斯优化的核心。它们近似评估成本高昂的目标函数，并提供指导优化过程的不确定性估计。

Bgolearn 支持多种代理模型，每种模型都有不同的优势和用例：

1. **高斯过程 (GP)** - 默认且最常见
2. **随机森林 (RF)** - 适用于离散/分类特征
3. **支持向量回归 (SVR)** - 对噪声具有鲁棒性
4. **多层感知器 (MLP)** - 神经网络方法
5. **AdaBoost** - 集成方法

## 高斯过程（GaussianProcess）

### 理论

高斯过程是贝叶斯优化的黄金标准。他们以有原则的方式提供预测和不确定性估计。

```python
from Bgolearn import BGOsampling

# Use Gaussian Process (default)
opt = BGOsampling.Bgolearn()
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Classifier='GaussianProcess'  # Explicit specification
)
```

### 优点

```{admonition} GP 优势
:class: tip
- **不确定性量化**：提供预测不确定性
- **理论基础**：完善的贝叶斯框架
- **平滑插值**：适用于连续函数
- **超参数很少**：相对容易调整
- **全局优化**：有利于寻找全局最优值
```

### 局限性

```{admonition} GP 局限性
:class: warning
- **计算成本**：使用训练数据进行 O(n³) 缩放
- **平滑度假设**：假设底层函数平滑
- **高维挑战**：与许多功能（> 20）作斗争
- **分类特征**：天生不适合离散变量
```

### 最佳用例

- **持续优化问题**
- **平滑目标函数**
- **中小型数据集**（<1000 个样本）
- **当不确定性很重要时**
- **材料性能优化**

### 示例：使用 GP 进行合金优化

```python
import numpy as np
import pandas as pd
import copy
from Bgolearn import BGOsampling

# Alloy composition optimization - use pandas DataFrame
data_matrix = pd.DataFrame([
    [2.0, 1.2, 0.5],  # Cu, Mg, Si
    [3.5, 0.8, 0.7],
    [1.8, 1.5, 0.3],
    [4.2, 0.9, 0.8]
], columns=['Cu', 'Mg', 'Si'])

strength_values = pd.Series([250, 280, 240, 290])  # MPa
measured_response = copy.deepcopy(strength_values)


virtual_samples = pd.DataFrame([
    [2.5, 1.0, 0.6],
    [3.0, 1.3, 0.4],
    [3.8, 0.9, 0.8]
], columns=['Cu', 'Mg', 'Si'])

# GP optimization
opt = BGOsampling.Bgolearn()
model = opt.fit(
    data_matrix=data_matrix,  # DataFrame
    Measured_response=strength_values,  # Series
    virtual_samples=virtual_samples,  # DataFrame
    Classifier='GaussianProcess',
    CV_test=2,  # 2-fold cross-validation
    Normalize=True
)

print("GP optimization completed!")
```

## 随机森林（RandomForest）

### 理论

随机森林构建多个决策树并对它们的预测进行平均。它特别适合处理离散特征和非线性关系。

```python
# Use Random Forest
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Classifier='RandomForest'
)
```

### 优点

```{admonition} RF 优势
:class: tip
- **处理离散特征**：对于分类变量来说是自然的
- **非线性关系**：捕获复杂模式
- **对异常值具有鲁棒性**：对噪声数据不太敏感
- **快速训练**：对于大型数据集有效
- **功能重要性**：提供功能排名
- **无假设**：不假设函数平滑度
```

### 局限性

```{admonition} RF 局限性
:class: warning
- **有限的不确定性**：不确定性量化较差
- **过度拟合风险**：小数据集可能会过度拟合
- **不连续**：创建逐步预测
- **超参数调优**：许多参数需要优化
```

### 最佳用例

- **离散/分类特征**
- **大型数据集**（>1000 个样本）
- **非光滑函数**
- **当稳健性很重要时**
- **混合变量类型**

### 示例：加工参数优化

```python
# Processing parameters with discrete levels - use pandas DataFrame
processing_data = pd.DataFrame([
    [450, 2, 1],    # Temperature, Time, Atmosphere (1=N2, 2=Ar, 3=Air)
    [500, 4, 2],
    [550, 6, 3],
    [480, 3, 1]
], columns=['Temperature', 'Time', 'Atmosphere'])

hardness_values = pd.Series([180, 220, 250, 200])

virtual_processing = pd.DataFrame([
    [475, 3.5, 1],
    [525, 4.5, 2],
    [490, 2.5, 3]
], columns=['Temperature', 'Time', 'Atmosphere'])

# Random Forest for mixed variables
model = opt.fit(
    data_matrix=processing_data,  # DataFrame
    Measured_response=hardness_values,  # Series
    virtual_samples=virtual_processing,  # DataFrame
    Classifier='RandomForest',
    CV_test='LOOCV'  # Leave-one-out cross-validation
)

print("Random Forest optimization completed!")
```

## 支持向量回归 (SVR)

### 理论

SVR 使用支持向量机进行回归，找到一个与目标偏差最多 ε 的函数，同时尽可能平坦。

```python
# Use SVR
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Classifier='SVR'
)
```

### 优点

```{admonition} SVR 优势
:class: tip
- **抗噪声能力强**：很好地处理噪声数据
- **高维**：适用于许多功能
- **内核灵活性**：可用不同的内核功能
- **稀疏解决方案**：仅使用支持向量
- **正则化**：内置过拟合保护
```

### 局限性

```{admonition} SVR 局限性
:class: warning
- **参数灵敏度**：需要仔细调整
- **无不确定性**：不提供预测不确定性
- **内核选择**：选择正确的内核至关重要
- **计算成本**：对于大型数据集可能会很慢
```

### 最佳用例

- **噪声数据**
- **高维问题**
- **当稳健性至关重要时**
- **非线性关系**

### 示例：高维优化

```python
# High-dimensional alloy with many elements - use pandas DataFrame
element_names = ['Al', 'Cu', 'Mg', 'Si', 'Fe', 'Mn', 'Cr', 'Zn']
high_dim_data = pd.DataFrame(
    np.random.random((20, 8)),  # 8 alloying elements
    columns=element_names
)
high_dim_response = pd.Series(np.random.random(20) * 100 + 200)

high_dim_virtual = pd.DataFrame(
    np.random.random((50, 8)),
    columns=element_names
)

# SVR for high-dimensional problem
model = opt.fit(
    data_matrix=high_dim_data,  # DataFrame
    Measured_response=high_dim_response,  # Series
    virtual_samples=high_dim_virtual,  # DataFrame
    Classifier='SVR',
    Normalize=True
)

print("SVR optimization completed!")
```

## 多层感知器 (MLP)

### 理论

MLP 是一种具有多个隐藏层的神经网络，可以逼近复杂的非线性函数。

```python
# Use MLP
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Classifier='MLP'
)
```

### 优点

```{admonition} MLP 优势
:class: tip
- **通用逼近器**：可以逼近任何连续函数
- **非线性建模**：非常适合复杂关系
- **Scalable**: Can handle large datasets
- **灵活的架构**：可定制的网络结构
```

### 局限性

```{admonition} MLP 局限性
:class: warning
- **需要大数据**：需要大量的训练数据
- **超参数调优**：许多参数需要优化
- **过度拟合风险**：很容易过度拟合小数据集
- **No Uncertainty**: Doesn't provide uncertainty estimates
- **训练不稳定**：可能对初始化敏感
```

### 最佳用例

- **大型数据集**（> 500 个样本）
- **复杂的非线性关系**
- **当有足够的数据可用时**
- **模式识别任务**

## 阿达助推器

### 理论

AdaBoost（自适应增强）结合了多个弱学习器来创建一个强大的预测器。

```python
# Use AdaBoost
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Classifier='AdaBoost'
)
```

### 优点

```{admonition} AdaBoost 优势
:class: tip
- **集成方法**：组合多个模型
- **自适应**：专注于困难的例子
- **减少偏差**：通常可以提高预测准确性
- **多功能**：适用于不同的基础学习者
```

### 局限性

```{admonition} AdaBoost 局限性
:class: warning
- **对噪声敏感**：可能会过度拟合噪声数据
- **计算成本**：比单一模型慢
- **参数调整**：需要仔细调整
- **有限的不确定性**：不确定性量化较差
```

## 选型指南

### 决策树

```
Start Here
    ↓
Do you have uncertainty requirements?
    ↓ Yes                    ↓ No
Gaussian Process         What type of features?
    ↓                        ↓
Is data smooth?         Continuous → SVR/MLP
    ↓ Yes    ↓ No           ↓
    GP    Random Forest   Discrete → Random Forest
                             ↓
                        Large dataset?
                             ↓ Yes    ↓ No
                            MLP    Random Forest
```

### 详细的选择标准

```{list-table} Model Selection Guide
:header-rows: 1
:name: model-selection

* - 标准
  - 高斯过程
  - 随机森林
  - 支持向量机
  - 多层线性规划
  - 阿达助推器
* - **数据大小**
  - <1000
  - >500
  - >500
  - >500
  - >500
* - **功能类型**
  - 连续的
  - 混合
  - 连续的
  - 连续的
  - 混合
* - **不确定性**
  - 出色的
  - 贫穷的
  - 没有任何
  - 没有任何
  - 贫穷的
* - **噪音容忍度**
  - 中等的
  - 高的
  - 高的
  - 中等的
  - 低的
* - **Training Speed**
  - 慢的
  - 快速地
  - 中等的
  - 慢的
  - 中等的
* - **预测速度**
  - 快速地
  - 快速地
  - 快速地
  - 快速地
  - 中等的
```

## 实际例子

### 型号对比

```python
# Create test data for model comparison
data_matrix = pd.DataFrame([
    [2.0, 1.2, 0.5],
    [3.5, 0.8, 0.7],
    [1.8, 1.5, 0.3],
    [4.2, 0.9, 0.8],
    [2.8, 1.1, 0.6]
], columns=['Cu', 'Mg', 'Si'])

measured_response = pd.Series([250, 280, 240, 290, 265])

virtual_samples = pd.DataFrame([
    [2.5, 1.0, 0.6],
    [3.0, 1.3, 0.4],
    [3.8, 0.9, 0.8]
], columns=['Cu', 'Mg', 'Si'])

# Compare all models on the same problem
models = ['GaussianProcess', 'RandomForest', 'SVR', 'MLP', 'AdaBoost']
results = {}

for model_name in models:
    print(f"Testing {model_name}...")

    try:
        model = opt.fit(
            data_matrix=data_matrix,  # DataFrame
            Measured_response=measured_response,  # Series
            virtual_samples=virtual_samples,  # DataFrame
            Classifier=model_name,
            CV_test=5,  # 5-fold cross-validation
            Normalize=True
        )
        
        # Get results using EI
        ei_values, recommended_points = model.EI()
        recommended_point = recommended_points[0]

        # Get model performance metrics (if available)
        try:
            # Access cross-validation score if available
            cv_score = getattr(model, 'cv_score_', 0.0)
        except:
            cv_score = 0.0

        results[model_name] = {
            'recommended_point': recommended_point,
            'ei_max': np.max(ei_values),
            'cv_score': cv_score,
            'success': True
        }

        print(f"  Success: EI max = {np.max(ei_values):.3f}")
        
    except Exception as e:
        print(f"  Failed: {str(e)}")
        results[model_name] = {'success': False, 'error': str(e)}

# Display comparison
print("\nModel Performance Comparison:")
print("-" * 50)
for model, result in results.items():
    if result['success']:
        print(f"{model:15s}: Success - EI max = {result['ei_max']:.3f}")
    else:
        print(f"{model:15s}: Failed - {result.get('error', 'Unknown error')}")
```

### 超参数调优

```python
# Hyperparameter tuning in Bgolearn
import numpy as np
import pandas as pd
from Bgolearn import BGOsampling

# Create test data
data_matrix = pd.DataFrame([
    [2.0, 1.2, 0.5, 450],  # Cu, Mg, Si, Temperature
    [3.5, 0.8, 0.7, 500],
    [1.8, 1.5, 0.3, 480],
    [4.2, 0.9, 0.8, 520],
    [2.8, 1.1, 0.6, 490],
    [3.2, 1.3, 0.4, 510],
    [2.5, 0.9, 0.7, 470],
    [3.8, 1.0, 0.5, 530]
], columns=['Cu', 'Mg', 'Si', 'Temperature'])

strength_values = pd.Series([250, 280, 240, 290, 265, 275, 245, 295])

virtual_samples = pd.DataFrame([
    [2.5, 1.0, 0.6, 485],
    [3.0, 1.3, 0.4, 505],
    [3.8, 0.9, 0.8, 515]
], columns=['Cu', 'Mg', 'Si', 'Temperature'])

# Hyperparameter tuning: compare different cross-validation settings
cv_settings = [3, 5, 10, 'LOOCV']
normalization_settings = [True, False]

best_score = -np.inf
best_config = None
results = {}

opt = BGOsampling.Bgolearn()

print("Starting hyperparameter tuning...")
print("=" * 50)

for cv_test in cv_settings:
    for normalize in normalization_settings:
        config_name = f"CV_{cv_test}_Norm_{normalize}"
        print(f"Testing configuration: {config_name}")

        try:
            model = opt.fit(
                data_matrix=data_matrix,
                Measured_response=strength_values,
                virtual_samples=virtual_samples,
                Classifier='GaussianProcess',
                CV_test=cv_test,
                Normalize=normalize,
                seed=42  # Ensure reproducibility
            )

            # Get EI values as performance metric
            ei_values, recommended_points = model.EI()
            max_ei = np.max(ei_values)

            # Calculate prediction quality
            predicted_mean = model.virtual_samples_mean
            predicted_std = model.virtual_samples_std

            # Combined score: EI max + inverse of prediction uncertainty
            uncertainty_score = 1.0 / (np.mean(predicted_std) + 1e-6)
            combined_score = max_ei + 0.1 * uncertainty_score

            results[config_name] = {
                'max_ei': max_ei,
                'mean_uncertainty': np.mean(predicted_std),
                'combined_score': combined_score,
                'recommended_point': recommended_points[0],
                'success': True
            }

            if combined_score > best_score:
                best_score = combined_score
                best_config = config_name

            print(f"  Success - EI max: {max_ei:.3f}, Combined score: {combined_score:.3f}")

        except Exception as e:
            print(f"  Failed: {str(e)}")
            results[config_name] = {'success': False, 'error': str(e)}

# Display tuning results
print(f"\n Hyperparameter tuning results:")
print("=" * 50)
print(f"{'Configuration':<20} {'EI Max':<10} {'Mean Uncertainty':<15} {'Combined Score':<15}")
print("-" * 60)

for config, result in results.items():
    if result['success']:
        print(f"{config:<20} {result['max_ei']:<10.3f} {result['mean_uncertainty']:<15.3f} {result['combined_score']:<15.3f}")
    else:
        print(f"{config:<20} {'Failed':<10} {'-':<15} {'-':<15}")

print(f"\n Best configuration: {best_config}")
print(f" Best score: {best_score:.3f}")

if best_config and results[best_config]['success']:
    best_point = results[best_config]['recommended_point']
    print(f" Next experiment point recommended by best configuration:")
    print(f"   Cu: {best_point[0]:.2f}%, Mg: {best_point[1]:.2f}%, Si: {best_point[2]:.2f}%, T: {best_point[3]:.0f}K")
```

## 高级主题

### 集成方法

通过组合多个代理模型来实现 Bgolearn 中的集成方法以提高性能：

```python
import numpy as np
import pandas as pd
from Bgolearn import BGOsampling

# Create test data
data_matrix = pd.DataFrame([
    [2.0, 1.2, 0.5],
    [3.5, 0.8, 0.7],
    [1.8, 1.5, 0.3],
    [4.2, 0.9, 0.8],
    [2.8, 1.1, 0.6],
    [3.2, 1.3, 0.4]
], columns=['Cu', 'Mg', 'Si'])

strength_values = pd.Series([250, 280, 240, 290, 265, 275])

virtual_samples = pd.DataFrame([
    [2.5, 1.0, 0.6],
    [3.0, 1.3, 0.4],
    [3.8, 0.9, 0.8],
    [2.2, 1.4, 0.5],
    [3.6, 0.7, 0.6]
], columns=['Cu', 'Mg', 'Si'])

# Define ensemble method class
class BgolearnEnsemble:
    """Bgolearn ensemble method implementation"""

    def __init__(self, model_types=['GaussianProcess', 'RandomForest', 'SVR']):
        self.model_types = model_types
        self.models = {}
        self.weights = None

    def fit(self, data_matrix, measured_response, virtual_samples, **kwargs):
        """Train all models"""
        print(" Training ensemble models...")

        opt = BGOsampling.Bgolearn()

        for model_type in self.model_types:
            print(f"  Training {model_type}...")
            try:
                model = opt.fit(
                    data_matrix=data_matrix,
                    Measured_response=measured_response,
                    virtual_samples=virtual_samples,
                    Classifier=model_type,
                    CV_test=5,
                    Normalize=True,
                    **kwargs
                )
                self.models[model_type] = model
                print(f"  {model_type} training successful")
            except Exception as e:
                print(f"  {model_type} training failed: {e}")

        # Calculate model weights (based on EI performance)
        self._calculate_weights()

    def _calculate_weights(self):
        """Calculate model weights based on EI performance"""
        if not self.models:
            return

        ei_scores = {}
        for name, model in self.models.items():
            try:
                ei_values, _ = model.EI()
                ei_scores[name] = np.max(ei_values)
            except:
                ei_scores[name] = 0.0

        # Normalize weights
        total_score = sum(ei_scores.values())
        if total_score > 0:
            self.weights = {name: score/total_score for name, score in ei_scores.items()}
        else:
            # Equal weights
            self.weights = {name: 1.0/len(self.models) for name in self.models.keys()}

        print(f"Model weights: {self.weights}")

    def ensemble_EI(self):
        """Ensemble expected improvement"""
        if not self.models:
            raise ValueError("No trained models available")

        ensemble_ei = None
        ensemble_points = None

        for name, model in self.models.items():
            try:
                ei_values, points = model.EI()
                weight = self.weights.get(name, 0.0)

                if ensemble_ei is None:
                    ensemble_ei = weight * ei_values
                    ensemble_points = points
                else:
                    ensemble_ei += weight * ei_values

            except Exception as e:
                print(f" {name} EI calculation failed: {e}")

        # Find the point corresponding to maximum EI
        if ensemble_ei is not None:
            max_idx = np.argmax(ensemble_ei)
            best_point = ensemble_points[max_idx:max_idx+1]  # Maintain dimensions
            return ensemble_ei, best_point
        else:
            raise ValueError("All model EI calculations failed")

    def ensemble_predictions(self):
        """Ensemble predictions"""
        predictions = {}
        uncertainties = {}

        for name, model in self.models.items():
            try:
                pred_mean = model.virtual_samples_mean
                pred_std = model.virtual_samples_std

                predictions[name] = pred_mean
                uncertainties[name] = pred_std

            except Exception as e:
                print(f" {name} prediction failed: {e}")

        if not predictions:
            raise ValueError("All model predictions failed")

        # Weighted average predictions
        ensemble_mean = np.zeros_like(list(predictions.values())[0])
        ensemble_std = np.zeros_like(list(uncertainties.values())[0])

        for name, pred in predictions.items():
            weight = self.weights.get(name, 0.0)
            ensemble_mean += weight * pred
            ensemble_std += weight * uncertainties[name]

        return ensemble_mean, ensemble_std

    def get_model_comparison(self):
        """Get model comparison results"""
        comparison = {}

        for name, model in self.models.items():
            try:
                ei_values, points = model.EI()
                pred_mean = model.virtual_samples_mean
                pred_std = model.virtual_samples_std

                comparison[name] = {
                    'max_ei': np.max(ei_values),
                    'mean_prediction': np.mean(pred_mean),
                    'mean_uncertainty': np.mean(pred_std),
                    'weight': self.weights.get(name, 0.0),
                    'recommended_point': points[0]
                }
            except Exception as e:
                comparison[name] = {'error': str(e)}

        return comparison

# Using ensemble methods
print("Starting ensemble method demonstration")
print("=" * 50)

# Create and train ensemble model
ensemble = BgolearnEnsemble(
    model_types=['GaussianProcess', 'RandomForest', 'SVR']
)

ensemble.fit(
    data_matrix=data_matrix,
    measured_response=strength_values,
    virtual_samples=virtual_samples,
    seed=42
)

# Get ensemble predictions
print("\n Ensemble prediction results:")
try:
    ensemble_mean, ensemble_std = ensemble.ensemble_predictions()
    print(f"Ensemble prediction mean: {ensemble_mean[:3]}")  # Show first 3
    print(f"Ensemble prediction std: {ensemble_std[:3]}")
except Exception as e:
    print(f"Ensemble prediction failed: {e}")

# Get ensemble EI
print("\n Ensemble Expected Improvement:")
try:
    ensemble_ei, ensemble_point = ensemble.ensemble_EI()
    max_ei_idx = np.argmax(ensemble_ei)
    print(f"Maximum ensemble EI value: {ensemble_ei[max_ei_idx]:.3f}")
    print(f"Recommended next experiment point: Cu={ensemble_point[0][0]:.2f}, Mg={ensemble_point[0][1]:.2f}, Si={ensemble_point[0][2]:.2f}")
except Exception as e:
    print(f"Ensemble EI calculation failed: {e}")

# Model comparison
print("\n Model performance comparison:")
comparison = ensemble.get_model_comparison()
print(f"{'Model':<15} {'Max EI':<10} {'Weight':<8} {'Mean Uncertainty':<15}")
print("-" * 55)

for name, metrics in comparison.items():
    if 'error' not in metrics:
        print(f"{name:<15} {metrics['max_ei']:<10.3f} {metrics['weight']:<8.3f} {metrics['mean_uncertainty']:<15.3f}")
    else:
        print(f"{name:<15} {'Failed':<10} {'-':<8} {'-':<15}")

print("\n Ensemble method advantages:")
print("  - Combines strengths of multiple models")
print("  - Improves prediction stability")
print("  - Reduces single model bias")
print("  - Automatic weight allocation")
```

### 迁移学习

使用预训练模型解决相关问题：

```python
# Conceptual transfer learning
def transfer_learning(source_model, target_data, target_response):
    """Transfer knowledge from source to target problem."""
    # Use source model as initialization
    # Fine-tune on target data
    pass
```

### 在线学习

用新数据更新模型：

```python
# Conceptual online learning
def online_update(model, new_x, new_y):
    """Update model with new observation."""
    # Add new data point
    # Retrain or update model incrementally
    pass
```

## 故障排除

### 常见问题

1. **简历分数不佳**
   - 尝试不同的型号
   - 检查数据质量
   - 增加训练数据
   - 调整标准化

2. **缓慢训练**
   - 对大数据使用随机森林
   - 减小虚拟空间大小
   - 考虑高维度的 SVR

3. **过度拟合**
   - 使用交叉验证
   - 降低模型复杂度
   - 添加更多训练数据

4. **不确定性较差**
   - 使用高斯过程
   - 增加引导迭代次数
   - 检查模型假设

### 性能优化

```python
# Tips for better performance
optimization_tips = {
    "Data preprocessing": "Normalize features, remove outliers",
    "Model selection": "Start with GP, try RF for discrete features",
    "Hyperparameters": "Use cross-validation for tuning",
    "Computational": "Reduce virtual space size for speed",
    "Validation": "Use CV_test=10 or 'LOOCV' for validation"
}
```

## 下一步

- **学习采集函数**：{doc}`acquisition_functions`
- **尝试优化策略**：{doc}`optimization_strategies`
- **举例练习**：{doc}`examples/single_objective`
- **探索多目标**：{doc}`multibgolearn`

```{seealso}
有关代理建模的更多信息：
- Rasmussen，C.E.“机器学习的高斯过程”
- Forrester, A.“通过代理建模进行工程设计”
- Queipo, N.V.“基于替代的分析和优化”
```
