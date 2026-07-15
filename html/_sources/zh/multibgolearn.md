# MultiBgolearn：多目标贝叶斯全局优化

```{note}
本页是 Bgolearn 手册的中文版本。
```

```{note}
MultiBgolearn 扩展了 Bgolearn 来处理多目标优化问题，从而能够同时优化材料设计中的多个竞争目标。
```

## 概述

MultiBgolearn 是专为多目标贝叶斯全局优化 (MOBO) 设计的 Python 软件包，专为材料设计量身定制。它通过同时优化多种材料属性来扩展 Bgolearn 封装的功能，使其非常适合在竞争目标之间进行权衡的现实应用。

```{admonition} 为什么需要多目标优化？
:class: tip
在材料设计中，我们经常需要同时优化多种性能：
- **强度与延展性**：更强的材料通常更脆
- **性能与成本**：更好的性能通常需要更高的成本
- **电导率与热稳定性**：高电导率材料可能热不稳定
- **耐腐蚀性与机械性能**：防腐处理可能会影响强度

MultiBgolearn 有助于找到这些竞争目标之间的最佳权衡。
```

## 主要特点

### 多种MOBO算法
- **预期超体积改进 (EHVI)**：最大化目标空间体积
- **q-噪声预期超体积改进 (qNEHVI)**：qNEHVI 扩展了 EHVI 以处理噪声观测和批量采集
- **改进概率 (PI)**：重点关注改进概率
- **置信上限 (UCB)**：平衡探索和利用

### 以材料为中心的设计
- 同时优化多种材料特性
- 灵活的目标处理（最大化/最小化）
- Bootstrap 不确定性量化
- 自动选型

### 灵活的替代模型
- 随机森林
- 梯度提升
- 支持向量回归 (SVR)
- 高斯过程
- 自动选型

## 安装

使用 pip 安装 MultiBgolearn：

```bash
pip install MultiBgolearn
```

或者一起安装两个包：

```bash
pip install Bgolearn MultiBgolearn
```

## 快速入门

以下是使用 MultiBgolearn 进行材料优化的简单示例：

```python
from MultiBgolearn import bgo

# Define your dataset and virtual space paths
dataset_path = './data/dataset.csv'
VS_path = './virtual_space/'

# Set the number of objectives (e.g., 3 for three-objective optimization)
object_num = 3

# Apply Multi-Objective Bayesian Global Optimization
VS_recommended, improvements, index = bgo.fit(
    dataset_path, 
    VS_path, 
    object_num, 
    max_search=True, 
    method='EHVI', 
    assign_model='GaussianProcess', 
    bootstrap=5
)

print(f"Recommended sample index: {index}")
print(f"Expected improvements: {improvements}")
```

## 数据格式

### 数据集格式
您的数据集应该是具有以下结构的 CSV 文件：

```csv
feature1,feature2,feature3,objective1,objective2,objective3
1.2,3.4,5.6,100.5,0.85,7.2
2.1,4.3,6.5,95.2,0.92,6.8
...
```

- **特征**：输入变量（成分、加工条件等）
- **目标**：要优化的目标属性（强度、延展性、成本等）

### 虚拟空间格式
虚拟空间包含优化的候选点：

```csv
feature1,feature2,feature3
1.5,3.2,5.8
2.3,4.1,6.2
...
```

## API参考

### 主要功能：`bgo.fit()`

```python
bgo.fit(dataset_path, VS_path, object_num, max_search=True, 
        method='EHVI', assign_model=False, bootstrap=5)
```

#### 参数

```{list-table} 参数
:header-rows: 1
:name: multibgo-parameters

* - 参数
  - 类型
  - 默认
  - 描述
* - `dataset_path`
  - str
  - 必需
  - 数据集 CSV 文件的路径
* - `VS_path`
  - str
  - 必需
  - 虚拟空间 CSV 文件的路径
* - `object_num`
  - int
  - 必需
  - 要优化的目标数量
* - `max_search`
  - bool
  - True
  - 最大化 (True) 还是最小化 (False)
* - `method`
  - str
  - 'EHVI'
  - 优化方法：'EHVI'、'qNEHVI'、'PI'、'UCB'
* - `assign_model`
  - str/bool
  - False
  - 代理模型或 False 用于自动选择
* - `bootstrap`
  - int
  - 5
  - 自举迭代次数
```

#### 返回值

```{list-table} 返回值
:header-rows: 1
:name: multibgo-returns

* - 变量
  - 类型
  - 描述
* - `VS_recommended`
  - array
  - 来自虚拟空间的推荐数据点
* - `improvements`
  - array
  - 每个目标的计算改进
* - `index`
  - int
  - 虚拟空间推荐点索引
```

## 优化方法

### 预期超容量改进 (EHVI)

EHVI 专注于最大化解决方案所主导的目标空间的体积。它对于具有 2-4 个目标的问题特别有效。

```python
# Use EHVI for balanced multi-objective optimization
VS_recommended, improvements, index = bgo.fit(
    dataset_path, VS_path, object_num=3,
    method='EHVI',
    assign_model='GaussianProcess'
)
```

**最适合：**
- 2-4个目标
- 帕累托前沿的平衡探索
- 当您想要最大化主导音量时


### q-噪声预期超容量改进 (qNEHVI)

qNEHVI 扩展了 EHVI 以处理噪声观测和批量采集，使其适用于具有测量不确定性和并行实验的现实场景。

```python
# Use qNEHVI for balanced multi-objective optimization
VS_recommended, improvements, index = bgo.fit(
    dataset_path, VS_path, object_num=3,
    method='qNEHVI',
    batch_size=3,  # Select 3 points simultaneously
    assign_model='GaussianProcess'
)
```

**最适合：**
- 测量结果具有显着的观察噪声或不确定性
- 多个实验可以并行进行
- 已知或估计的观测噪声水平



### 改进概率 (PI)

PI 选择对最知名的解决方案进行改进的可能性最高的点。

```python
# Use PI for improvement-focused optimization
VS_recommended, improvements, index = bgo.fit(
    dataset_path, VS_path, object_num=2,
    method='PI',
    assign_model='RandomForest'
)
```

**最适合：**
- 保守优化
- 当改进概率很重要时
- 以利用为重点的搜索

### 置信上限 (UCB)

UCB 探索具有最高置信上限的点，平衡探索和利用。

```python
# Use UCB for exploration-exploitation balance
VS_recommended, improvements, index = bgo.fit(
    dataset_path, VS_path, object_num=3,
    method='UCB',
    assign_model='GradientBoosting'
)
```

**最适合：**
- 吵闹的目标
- 当不确定性很重要时
- 平衡探索-利用

## 替代模型

### 自动选型

当`assign_model=False`、MultiBgolearn时自动选择最佳模型：

```python
# Automatic model selection
VS_recommended, improvements, index = bgo.fit(
    dataset_path, VS_path, object_num=3,
    assign_model=False  # Auto-select best model
)
```

### 手动选型

您可以显式指定代理模型：

```python
# Available models
models = [
    'RandomForest',
    'GradientBoosting', 
    'SVR',
    'GaussianProcess'
]

# Use specific model
VS_recommended, improvements, index = bgo.fit(
    dataset_path, VS_path, object_num=3,
    assign_model='GaussianProcess'
)
```

## Bootstrap 不确定性量化

MultiBgolearn 使用引导采样来量化预测中的不确定性：

```python
# Increase bootstrap iterations for better uncertainty estimation
VS_recommended, improvements, index = bgo.fit(
    dataset_path, VS_path, object_num=3,
    bootstrap=10  # More iterations = better uncertainty estimation
)
```

**引导程序的好处：**
- 稳健的不确定性量化
- 更好地处理噪声数据
- 提高模型可靠性
- 更有信心的推荐

## 实例：合金设计

让我们优化一个三目标合金设计问题：

```python
import pandas as pd
from MultiBgolearn import bgo

# Prepare data
# Dataset: composition features + 3 objectives (strength, ductility, cost)
dataset = pd.DataFrame({
    'Cu': [2.0, 3.5, 1.8, 4.2],
    'Mg': [1.2, 0.8, 1.5, 0.9],
    'Si': [0.5, 0.7, 0.3, 0.8],
    'Strength': [250, 280, 240, 290],    # Maximize
    'Ductility': [15, 12, 18, 10],      # Maximize  
    'Cost': [100, 120, 95, 130]         # Minimize (convert to maximize: -Cost)
})

# Convert cost to maximization problem
dataset['Cost'] = -dataset['Cost']

# Save dataset
dataset.to_csv('alloy_dataset.csv', index=False)

# Create virtual space (candidate compositions)
virtual_space = pd.DataFrame({
    'Cu': [2.5, 3.0, 3.8, 2.2, 4.0],
    'Mg': [1.0, 1.3, 0.9, 1.4, 1.1],
    'Si': [0.6, 0.4, 0.8, 0.5, 0.7]
})
virtual_space.to_csv('virtual_space.csv', index=False)

# Multi-objective optimization
VS_recommended, improvements, index = bgo.fit(
    'alloy_dataset.csv',
    'virtual_space.csv',
    object_num=3,
    max_search=True,
    method='EHVI',
    assign_model='GaussianProcess',
    bootstrap=5
)

print(f"Recommended alloy composition:")
print(f"Cu: {VS_recommended[0]:.2f}%")
print(f"Mg: {VS_recommended[1]:.2f}%") 
print(f"Si: {VS_recommended[2]:.2f}%")
print(f"Expected improvements: {improvements}")
```

## 最佳实践

### 1. 数据准备
- 确保足够的训练数据（每个目标 >10 个样本）
- 如果目标具有不同的尺度，则将其标准化
- 适当处理缺失值

### 2. 方法选择
- 使用 **EHVI** 进行 2-4 个目标的平衡探索
- 使用 **qNEHVI** 进行具有显着观测噪声或不确定性的测量
- 使用 **PI** 进行保守的、注重改进的搜索
- 使用 **UCB** 实现嘈杂的目标或探索需求

### 3. 型号选择
- 从自动模型选择开始
- 使用 **GaussianProcess** 实现平稳、连续的目标
- 将 **RandomForest** 用于离散或分类特征
- 使用 **GradientBoosting** 处理复杂的非线性关系

### 4. 引导程序设置
- 对于大多数问题，使用 5-10 次引导迭代
- 对于非常嘈杂的数据，增加到 20+
- 平衡计算时间与不确定性质量

## 故障排除

### 常见问题

1. **收敛问题**
   - 增加引导迭代次数
   - 尝试不同的代理模型
   - 检查数据质量和缩放比例

2. **糟糕的推荐**
   - 确保足够的训练数据
   - 验证目标定义（最大与最小）
   - 考虑数据预处理

3. **计算问题**
   - 减少引导迭代
   - 使用更简单的代理模型
   - 减小虚拟空间大小

### 性能技巧

- **数据大小**：保持虚拟空间可管理（<10,000 点）
- **目标**：EHVI 在 2-4 个目标下效果最佳
- **特征**：将特征标准化为相似的尺度
- **迭代**：从较少的引导迭代开始进行测试

## 下一步

- 了解 {doc}`multi_objective_concepts` 的理论背景
- 探索 {doc}`mobo_algorithms` 了解算法详细信息
- 尝试 {doc}`examples/multi_objective` 进行实践练习
- 了解 {doc}`pareto_optimization` 进行权衡分析

```{seealso}
欲了解更多信息：
- [MultiBgolearn GitHub 存储库](https://github.com/Bin-Cao/MultiBgolearn)
- [研究论文](https://bgolearn.netlify.app/)
- [材料设计示例](examples/materials_design.md)
```
