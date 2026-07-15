# MultiBgolearn 中的 MOBO 算法

```{note}
本页是 Bgolearn 手册的中文版本。
```

```{note}
本页提供了 MultiBgolearn 中实现的多目标贝叶斯优化算法的详细说明。
```

## 概述

MultiBgolearn 实现了三个主要的多目标采集功能，每个功能都有不同的优势和用例：

1. **预期超容量改进 (EHVI)** - 基于卷的优化
2. **q-Noisy 预期超体积改进 (qNEHVI)** - 基于体积的优化
3. **改进概率 (PI)** - 基于概率的改进
4. **置信上限 (UCB)** - 不确定性感知探索

## 预期超容量改进 (EHVI)

### 理论

EHVI 被认为是多目标贝叶斯优化的黄金标准。它最大化了超体积的预期改进，超体积是帕累托前沿主导的目标空间的体积。

```
EHVI(x) = E[HV(F ∪ {f(x)}) - HV(F)]
```

在哪里：
- `F`：当前帕累托前沿近似
- `HV(·)`：超成交量指标
- `f(x)`：x点的预测目标向量

### 数学细节

对于高斯过程代理模型，可以针对 2D 问题分析计算 EHVI，并针对更高维度进行近似：

```
EHVI(x) = ∫ max(0, HV_improvement) · p(f(x)) df(x)
```

其中 `p(f(x))` 是 GP 预测的多元高斯分布。

### 在 MultiBgolearn 中实现

```python
from MultiBgolearn import bgo

# Use EHVI for multi-objective optimization
VS_recommended, improvements, index = bgo.fit(
    dataset_path='data.csv',
    VS_path='virtual_space.csv', 
    object_num=3,
    method='EHVI',                    # Expected Hypervolume Improvement
    assign_model='GaussianProcess',   # Best for EHVI
    bootstrap=5
)
```

### 优点

```{admonition} EHVI 优势
:class: tip
- **理论上合理**：最大化明确定义的质量指标
- **帕累托服从**：尊重帕累托支配关系
- **平衡探索**：探索和开发之间的良好权衡
- **可扩展**：适用于 2-4 个目标
```

### 局限性

```{admonition} EHVI 局限性
:class: warning
- **计算成本**：对于超过 4 个目标而言成本昂贵
- **依赖参考点**：需要仔细选择参考点
- **需要近似**：精确计算仅适用于 2D
```

### 最佳用例

- **2-4 个目标**：最佳性能范围
- **平衡探索**：当您想要全面的帕累托前沿时
- **质量焦点**：当超容量成为重要指标时
- **材料设计**：合金优化、加工参数

### 示例：使用 EHVI 进行合金设计

```python
import pandas as pd
from MultiBgolearn import bgo

# Prepare alloy dataset
# Features: Cu, Mg, Si content (%)
# Objectives: Strength (MPa), Ductility (%), Cost ($/kg)
dataset = pd.DataFrame({
    'Cu': [2.0, 3.5, 1.8, 4.2, 2.8],
    'Mg': [1.2, 0.8, 1.5, 0.9, 1.1], 
    'Si': [0.5, 0.7, 0.3, 0.8, 0.6],
    'Strength': [250, 280, 240, 290, 265],
    'Ductility': [15, 12, 18, 10, 14],
    'Cost': [-100, -120, -95, -130, -110]  # Negative for maximization
})

dataset.to_csv('alloy_data.csv', index=False)

# Virtual space
virtual_space = pd.DataFrame({
    'Cu': [2.5, 3.0, 3.8, 2.2],
    'Mg': [1.0, 1.3, 0.9, 1.4],
    'Si': [0.6, 0.4, 0.8, 0.5]
})
virtual_space.to_csv('virtual_alloys.csv', index=False)

# EHVI optimization
recommended, improvements, idx = bgo.fit(
    'alloy_data.csv',
    'virtual_alloys.csv',
    object_num=3,
    method='EHVI',
    assign_model='GaussianProcess',
    bootstrap=10
)

print(f"Recommended alloy: Cu={recommended[0]:.1f}%, Mg={recommended[1]:.1f}%, Si={recommended[2]:.1f}%")
print(f"Expected improvements: Strength={improvements[0]:.1f}, Ductility={improvements[1]:.1f}, Cost={improvements[2]:.1f}")
```
## q-噪声预期超容量改进 (qNEHVI)

### 理论

qNEHVI 扩展了 EHVI 来处理噪声观测和批量采集，使其成为具有测量不确定性和并行实验的现实世界多目标优化的最先进方法。

```
qNEHVI(X_q) = E_y[E_f[HV(P(y ∪ f_q)) - HV(P(y))]]
```


在哪里：
- `X_q = {x_1, ..., x_q}`：q个候选点的批次
- `y`：现有数据的噪声观测值
- `f_q`：候选批次的预测目标
- `P(·)`：帕累托前沿算子
- 内心期望：关于 GP 后验预测
- 外部期望：过度观察噪声

### 数学细节

qNEHVI 在采集函数中明确模拟观测噪声：

```
y_i(x) = f_i(x) + ε_i, ε_i ~ N(0, σ²_obs,i)
```

采集函数通过嵌套蒙特卡罗采样来近似：
```
qNEHVI(X_q) ≈ (1/S₁S₂) Σ Σ max(0, HV(P(y^(s₁) ∪ f_q^(s₂))) - HV(P(y^(s₁))))
```



### 示例：使用 qNEHVI 进行合金设计

```python
import pandas as pd
from MultiBgolearn import bgo

# Prepare alloy dataset
# Features: Cu, Mg, Si content (%)
# Objectives: Strength (MPa), Ductility (%), Cost ($/kg)
dataset = pd.DataFrame({
    'Cu': [2.0, 3.5, 1.8, 4.2, 2.8],
    'Mg': [1.2, 0.8, 1.5, 0.9, 1.1], 
    'Si': [0.5, 0.7, 0.3, 0.8, 0.6],
    'Strength': [250, 280, 240, 290, 265],
    'Ductility': [15, 12, 18, 10, 14],
    'Cost': [-100, -120, -95, -130, -110]  # Negative for maximization
})

dataset.to_csv('alloy_data.csv', index=False)

# Virtual space
virtual_space = pd.DataFrame({
    'Cu': [2.5, 3.0, 3.8, 2.2],
    'Mg': [1.0, 1.3, 0.9, 1.4],
    'Si': [0.6, 0.4, 0.8, 0.5]
})
virtual_space.to_csv('virtual_alloys.csv', index=False)

# EHVI optimization
recommended, improvements, idx = bgo.fit(
    'alloy_data.csv',
    'virtual_alloys.csv',
    object_num=3,
    method='qNEHVI',
    assign_model='GaussianProcess',
    batch_size=3,
    bootstrap=5
)

print(f"Recommended alloy: Cu={recommended[0]:.1f}%, Mg={recommended[1]:.1f}%, Si={recommended[2]:.1f}%")
print(f"qNEHVI improvements: Strength={improvements[0]:.1f}, Ductility={improvements[1]:.1f}, Cost={improvements[2]:.1f}")
```

## 改进概率 (PI)

### 理论

多目标 PI 计算候选点将改进当前 Pareto 前沿中至少一个目标的概率。

```
PI(x) = P(f(x) dominates at least one point in F)
```

对于高斯过程预测：

```
PI(x) = P(∃ y ∈ F : f(x) ≻ y)
```

其中 `≻` 表示帕累托优势。

### 执行

```python
# Use PI for improvement-focused optimization
VS_recommended, improvements, index = bgo.fit(
    dataset_path='data.csv',
    VS_path='virtual_space.csv',
    object_num=2,
    method='PI',                      # Probability of Improvement
    assign_model='RandomForest',      # Good for discrete problems
    bootstrap=5
)
```

### 计算详情

对于每个候选点 x，PI 计算如下：

1. **从 GP 后验生成样本**：`f_samples ~ GP(x)`
2. **检查每个样本相对于帕累托前沿的优势**
3. **计算概率**作为主导样本的分数

```python
# Pseudo-code for PI calculation
def calculate_PI(x, pareto_front, gp_model, n_samples=1000):
    # Sample from GP posterior
    f_samples = gp_model.sample_posterior(x, n_samples)
    
    # Check dominance for each sample
    dominates_count = 0
    for sample in f_samples:
        if any(dominates(sample, pf_point) for pf_point in pareto_front):
            dominates_count += 1
    
    return dominates_count / n_samples
```

### 优点

```{admonition} PI 优势
:class: tip
- **直观**：易于理解和解释
- **保守**：关注可能的改进
- **计算速度快**：计算效率相对较高
- **稳健**：适用于嘈杂的目标
```

### 局限性

```{admonition} PI 局限性
:class: warning
- **利用偏差**：可能探索得不够
- **局部最优**：可能会陷入局部区域
- **忽略幅度**：不考虑改进大小
```

### 最佳用例

- **保守优化**：当您想要可靠的改进时
- **嘈杂的目标**：当测量具有高度不确定性时
- **预算有限**：当您的评估机会很少时
- **开发重点**：当您想要改进已知的良好区域时

### 示例：加工参数优化

```python
# Heat treatment optimization
# Features: Temperature (°C), Time (hours), Cooling rate (°C/min)
# Objectives: Hardness (HV), Toughness (J), Energy cost (kWh)

dataset = pd.DataFrame({
    'Temperature': [450, 500, 550, 480, 520],
    'Time': [2, 4, 6, 3, 5],
    'Cooling_rate': [10, 20, 15, 25, 12],
    'Hardness': [180, 220, 250, 200, 235],
    'Toughness': [45, 35, 25, 40, 30],
    'Energy_cost': [-50, -80, -120, -65, -95]  # Negative for maximization
})

dataset.to_csv('heat_treatment_data.csv', index=False)

# Virtual processing conditions
virtual_conditions = pd.DataFrame({
    'Temperature': [475, 525, 490, 510],
    'Time': [3.5, 5.5, 2.5, 4.5],
    'Cooling_rate': [18, 22, 14, 16]
})
virtual_conditions.to_csv('virtual_conditions.csv', index=False)

# PI optimization
recommended, improvements, idx = bgo.fit(
    'heat_treatment_data.csv',
    'virtual_conditions.csv',
    object_num=3,
    method='PI',
    assign_model='GradientBoosting',
    bootstrap=8
)

print(f"Recommended conditions: T={recommended[0]:.0f}°C, t={recommended[1]:.1f}h, CR={recommended[2]:.0f}°C/min")
```

## 置信上限 (UCB)

### 理论

多目标 UCB 通过使用每个目标的置信界限来平衡利用（平均预测）和探索（不确定性）。

```
UCB(x) = μ(x) + β·σ(x)
```

在哪里：
- `μ(x)`：平均预测向量
- `σ(x)`：标准差向量  
- `β`：探索参数

### 多目标 UCB 变体

#### 1. 组件级 UCB
分别将 UCB 应用于每个目标：

```
UCB_i(x) = μ_i(x) + β·σ_i(x)  for i = 1, ..., m
```

#### 2.基于超容量的UCB
使用 UCB 值计算超体积：

```
UCB_HV(x) = HV([UCB_1(x), UCB_2(x), ..., UCB_m(x)])
```

#### 3. 标化UCB
将目标与权重结合起来：

```
UCB_scalar(x) = Σ w_i · UCB_i(x)
```

### 执行

```python
# Use UCB for exploration-focused optimization
VS_recommended, improvements, index = bgo.fit(
    dataset_path='data.csv',
    VS_path='virtual_space.csv',
    object_num=3,
    method='UCB',                     # Upper Confidence Bound
    assign_model='SVR',               # Good for high-dimensional problems
    bootstrap=5
)
```

### 参数选择

探索参数 β 控制探索与利用的权衡：

```python
# β selection guidelines
beta_values = {
    "Conservative (exploitation)": 0.5,
    "Balanced": 1.0,
    "Aggressive (exploration)": 2.0,
    "Very aggressive": 3.0
}
```

### 优点

```{admonition} UCB 优势
:class: tip
- **不确定性意识**：明确考虑预测不确定性
- **可调**：β参数允许探索控制
- **稳健**：适用于噪声数据
- **可扩展**：对于许多目标都有效
```

### 局限性

```{admonition} UCB 局限性
:class: warning
- **参数调整**：需要选择β
- **过度探索**：可能在高 β 的情况下探索过多
- **取决于模型**：性能取决于不确定性估计
```

### 最佳用例

- **嘈杂的目标**：当测量具有显着的不确定性时
- **探索需求**：当你想要探索未知区域时
- **高维度**：当您有很多目标时 (>4)
- **不确定性量化**：当预测置信度很重要时

### 示例：多属性陶瓷设计

```python
# Ceramic material optimization
# Features: Al2O3, SiO2, MgO content (%)
# Objectives: Strength (MPa), Thermal conductivity (W/mK), Thermal shock resistance

dataset = pd.DataFrame({
    'Al2O3': [85, 90, 80, 95, 88],
    'SiO2': [10, 5, 15, 3, 8],
    'MgO': [5, 5, 5, 2, 4],
    'Strength': [300, 350, 280, 380, 320],
    'Thermal_conductivity': [25, 30, 20, 35, 28],
    'Thermal_shock': [8, 6, 10, 5, 7]
})

dataset.to_csv('ceramic_data.csv', index=False)

# Virtual compositions
virtual_ceramics = pd.DataFrame({
    'Al2O3': [87, 92, 83, 89],
    'SiO2': [8, 4, 12, 7],
    'MgO': [5, 4, 5, 4]
})
virtual_ceramics.to_csv('virtual_ceramics.csv', index=False)

# UCB optimization with high exploration
recommended, improvements, idx = bgo.fit(
    'ceramic_data.csv',
    'virtual_ceramics.csv',
    object_num=3,
    method='UCB',
    assign_model='GaussianProcess',
    bootstrap=12  # Higher bootstrap for better uncertainty
)

print(f"Recommended ceramic: Al2O3={recommended[0]:.1f}%, SiO2={recommended[1]:.1f}%, MgO={recommended[2]:.1f}%")
```

## 算法比较

### 性能特点

```{list-table} Algorithm Comparison
:header-rows: 1
:name: mobo-comparison

* - 算法
  - 最适合
  - 目标
  - 勘探
  - 计算
* - **EHVI**
  - 均衡优化
  - 2-4
  - 缓和
  - 高的
* - **qNEHVI**
  - 均衡优化
  - 2-4
  - 缓和
  - 高变化
* - **PI**
  - 保守改进
  - 2-6
  - 低的
  - 中等的
* - **UCB**
  - 不确定/噪音数据
  - 2-8+
  - 高的
  - 低的
```

### 选择指南

#### 在以下情况下选择 EHVI：
- 您有 2-4 个目标
- 您想要全面的帕累托前沿
- 计算资源可用
- 超容量是重要指标


#### 在以下情况下选择 qNEHVI：
- 您有 2-4 个目标
- 测量结果具有显着的观察噪声或不确定性
- 多个实验可以并行进行



#### 在以下情况下选择 PI：
- 您想要可靠的改进
- 目标有噪音
- 评估预算有限
- 首选保守方法

#### 在以下情况下选择 UCB：
- 您有 >4 个目标
- 测量不确定度高
- 需要探索未知区域
- 想要可调探索

## 高级主题

### 约束处理

所有算法都可以通过惩罚方法处理约束：

```python
# Add constraints to optimization
def constraint_penalty(x):
    penalty = 0
    # Composition constraint: sum ≤ 100%
    if sum(x[:3]) > 100:
        penalty += 1000 * (sum(x[:3]) - 100)
    # Temperature constraint: 400 ≤ T ≤ 600
    if x[3] < 400 or x[3] > 600:
        penalty += 1000 * abs(x[3] - np.clip(x[3], 400, 600))
    return penalty
```

### 批量优化

对于并行实验，修改采集函数：

```python
# Batch EHVI (simplified concept)
def batch_EHVI(X_batch, pareto_front, gp_models):
    # Compute joint improvement for entire batch
    # More complex than single-point EHVI
    pass
```

### 动态目标

处理随时间变化的目标：

```python
# Update objectives based on new requirements
def update_objectives(iteration):
    if iteration < 10:
        return ['strength', 'ductility']
    else:
        return ['strength', 'ductility', 'cost']  # Add cost later
```

## 实施技巧

### 数据预处理

```python
# Normalize objectives to similar scales
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
objectives_normalized = scaler.fit_transform(objectives)
```

### 选型

```python
# Model recommendations by algorithm
model_recommendations = {
    'EHVI': 'GaussianProcess',      # Best uncertainty estimates
    'qNEHVI': 'GaussianProcess',      # Best uncertainty estimates
    'PI': 'RandomForest',           # Robust to noise
    'UCB': 'GaussianProcess'        # Good uncertainty quantification
}
```

### 引导程序设置

```python
# Bootstrap recommendations by algorithm
bootstrap_settings = {
    'EHVI': 5,   # Moderate bootstrap for balance
    'qNEHVI': 5,   # Moderate bootstrap for balance
    'PI': 8,     # Higher for noise robustness  
    'UCB': 10    # Highest for uncertainty estimation
}
```

## 故障排除

### 常见问题

1. **收敛性差**
   - 增加引导迭代次数
   - 尝试不同的代理模型
   - 检查目标缩放

2. **性能缓慢**
   - 减小虚拟空间大小
   - 使用更简单的模型（RandomForest 与 GaussianProcess）
   - 减少引导迭代

3. **意外的推荐**
   - 验证目标定义（最大与最小）
   - 检查数据预处理
   - 验证约束处理

### 性能优化

```python
# Speed up optimization
optimization_tips = {
    "Reduce virtual space": "Keep <5000 candidates",
    "Simplify models": "Use RandomForest for speed",
    "Parallel bootstrap": "Use multiprocessing",
    "Cache computations": "Store expensive calculations"
}
```

## 下一步

现在您已经了解了 MOBO 算法：

1. **举例练习**：{doc}`examples/multi_objective`
2. **学习帕累托分析**：{doc}`pareto_optimization`
3. **尝试实际应用**：{doc}`examples/materials_design`
4. **探索高级功能**：{doc}`examples/advanced_usage`

```{seealso}
有关算法详细信息：
- Emmerich, M.“单目标和多目标进化优化”
- 琼斯，D.R. “昂贵的黑盒功能的高效全局优化”
- Knowles, J.“ParEGO：一种具有在线景观近似的混合算法”
- Samuel, D.“多个噪声目标的并行贝叶斯优化与预期的超体积改进”
```
