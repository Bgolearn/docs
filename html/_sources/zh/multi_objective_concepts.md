# 多目标优化概念

```{note}
本页是 Bgolearn 手册的中文版本。
```

```{note}
本页介绍了多目标优化的基本概念以及它们如何应用于材料设计问题。
```

## 什么是多目标优化？

多目标优化 (MOO) 处理的是我们需要同时优化多个经常相互冲突的目标的问题。与寻求一个最佳解决方案的单目标优化不同，MOO 通常会产生一组权衡解决方案，称为 **帕累托前沿**。

```{admonition} 真实示例
:class: tip
考虑设计一种新合金：
- **目标 1**：最大化力量（越高越好）
- **目标 2**：最大限度地提高延展性（越高越好）  
- **目标 3**：最小化成本（越低越好）

这些目标经常相互冲突：强度更高的合金可能更脆（延展性较差）并且更昂贵。
```

## 关键概念

### 帕累托优势

如果一种解决方案在至少一个目标上更好并且在所有其他目标上都没有更差，则该解决方案**支配**另一个解决方案。

```python
# Example: Two alloy compositions
Alloy_A = [Strength=250, Ductility=15, Cost=100]
Alloy_B = [Strength=240, Ductility=15, Cost=100]

# Alloy A dominates Alloy B because:
# - Strength: 250 > 240 (better)
# - Ductility: 15 = 15 (equal)
# - Cost: 100 = 100 (equal)
```

### 帕累托前沿

**帕累托前沿**（或帕累托前沿）是所有非支配解的集合。这些代表了目标之间的最佳可能权衡。

```
Strength vs. Ductility Trade-off:

Ductility
    ^
    |     * (Pareto optimal)
    |   *   * (Pareto optimal)
    | *       * (Pareto optimal)
    |           * (Pareto optimal)
    |             *
    +----------------> Strength
    
Points on the curve are Pareto optimal.
Points below the curve are dominated.
```

### 帕累托最优

如果没有其他解决方案占主导地位，则解决方案是**帕累托最优**。多目标优化的目标是找到帕累托前沿。

## 数学公式

### 一般多目标问题

```
minimize/maximize: f(x) = [f₁(x), f₂(x), ..., fₘ(x)]
subject to: g(x) ≤ 0
           h(x) = 0
           x ∈ X
```

在哪里：
- `f(x)`：m 个目标函数的向量
- `g(x)`：不等式约束
- `h(x)`：平等约束
- `X`：决策变量空间

### 材料设计实例

对于合金优化：

```
maximize: f₁(x) = Strength(Cu, Mg, Si, T_aging)
maximize: f₂(x) = Ductility(Cu, Mg, Si, T_aging)
minimize: f₃(x) = Cost(Cu, Mg, Si, T_aging)

subject to: Cu + Mg + Si ≤ 10%  (composition constraint)
           150 ≤ T_aging ≤ 200   (temperature constraint)
           Cu, Mg, Si ≥ 0        (non-negativity)
```

## 多目标与单目标

```{list-table} Comparison
:header-rows: 1
:name: mo-vs-so

* - 方面
  - 单一目标
  - 多目标
* - **解决方案**
  - 一个最佳点
  - 一套权衡解决方案（帕累托前沿）
* - **决策**
  - 自动的
  - 需要偏好信息
* - **复杂性**
  - 降低
  - 更高（维数诅咒）
* - **可视化**
  - 简单（一维或二维）
  - 困难（高维）
* - **算法**
  - 信誉卓著
  - 更复杂、更新的领域
```

## 多目标优化的挑战

### 1. 目标相互冲突

大多数现实世界的问题都涉及权衡：

```python
# Materials examples
conflicts = {
    "Strength vs. Ductility": "Stronger materials are often more brittle",
    "Performance vs. Cost": "Better performance usually costs more",
    "Conductivity vs. Stability": "High conductivity may reduce thermal stability",
    "Hardness vs. Toughness": "Harder materials may be less tough"
}
```

### 2.维度诅咒

随着目标数量的增加，问题变得更加复杂：

- **2-3 个目标**：可管理、良好的可视化
- **4-6 个目标**：具有挑战性、可视化有限
- **>6 个目标**：非常困难，多目标优化

### 3. 方案选择

Pareto前沿提供了多种解决方案，但我们经常需要选择一种：

- **先验**：在优化之前定义偏好
- **后验**：优化后从帕累托前沿选择
- **交互式**：在优化期间迭代地细化偏好

## 多目标贝叶斯优化 (MOBO)

MOBO 通过以下方式扩展贝叶斯优化以处理多个目标：

1. **替代模型**：为每个目标构建单独的模型
2. **多目标采集**：使用考虑所有目标的采集函数
3. **帕累托前沿近似**：迭代改进帕累托前沿估计

### 主要优势

```{admonition} 为什么材料研究需要 MOBO？
:class: tip
- **昂贵的实验**：材料测试成本高昂且耗时
- **多重属性**：材料具有许多重要属性
- **权衡理解**：需要了解属性之间的关系
- **高效探索**：用最少的实验找到帕累托前沿
```

## MOBO采集功能

### 预期超容量改进 (EHVI)

EHVI 最大化超体积（主导目标空间的体积）的预期改进：

```
EHVI(x) = E[HV(F ∪ {f(x)}) - HV(F)]
```

在哪里：
- `F`：当前帕累托前沿近似
- `HV`：超成交量指标
- `f(x)`：x点的目标向量

**最适合**：2-4 个目标，平衡探索



### q-噪声预期超容量改进 (qNEHVI)

qNEHVI 扩展了 EHVI 以处理噪声观测和批量采集，使其适用于具有测量不确定性和并行实验的现实场景：

```
qNEHVI(x) = E_y[E[HV(F ∪ {f(x)}) - HV(F)]]
```

在哪里：
- `y`：有噪音的观察结果
- `HV`：超成交量指标
- `f(x)`：x点的目标向量

**最适合**：测量结果具有显着的观察噪声



### 改进概率 (PI)

多目标 PI 考虑每个目标的改进概率：

```
PI(x) = P(f(x) dominates at least one point in F)
```

**最适合**：保守优化，注重利用

### 置信上限 (UCB)

多目标 UCB 平衡平均预测和不确定性：

```
UCB(x) = μ(x) + β·σ(x)
```

单独或组合应用于每个目标。

**最适合**：吵闹的目标，专注于探索

## 超成交量指标

超体积是帕累托前沿的关键质量指标：

```
HV(F) = Volume of space dominated by F
```

### 特性
- **单调**：更大的超体积 = 更好的帕累托前沿
- **符合帕累托**：尊重帕累托优势
- **依赖参考点**：需要参考点

### 计算示例

对于 2D 目标（最大化两者）：

```python
# Pareto front points
pareto_points = [(3, 1), (2, 2), (1, 3)]
reference_point = (0, 0)

# Hypervolume = sum of dominated rectangles
# Rectangle 1: (3-0) × (1-0) = 3
# Rectangle 2: (2-0) × (2-1) = 2  
# Rectangle 3: (1-0) × (3-2) = 1
# Total HV = 3 + 2 + 1 = 6
```

## 实际考虑

### 目标缩放

不同的目标可能有不同的尺度：

```python
# Before scaling
objectives = {
    "Strength": [200, 300, 250],      # MPa
    "Ductility": [10, 20, 15],       # %
    "Cost": [50, 100, 75]            # $/kg
}

# After normalization (0-1 scale)
normalized = {
    "Strength": [0.0, 1.0, 0.5],
    "Ductility": [0.0, 1.0, 0.5], 
    "Cost": [0.0, 1.0, 0.5]
}
```

### 参考点选择

对于超体积计算，请仔细选择参考点：

```python
# Good reference point (slightly worse than worst known values)
reference_point = [min_strength - 0.1*range_strength,
                  min_ductility - 0.1*range_ductility,
                  max_cost + 0.1*range_cost]  # Note: max for minimization
```

### 目标转变

将最小化转换为最大化：

```python
# Original objectives
strength = 250      # maximize
ductility = 15      # maximize  
cost = 100          # minimize

# Transformed for maximization
objectives = [strength, ductility, -cost]  # Negate cost
```

## 材料设计应用

### 合金成分优化

```python
# Objectives for structural alloys
objectives = [
    "Yield Strength",     # Maximize
    "Ultimate Strength",  # Maximize
    "Elongation",        # Maximize
    "Fatigue Life",      # Maximize
    "Cost",              # Minimize
    "Density"            # Minimize (for aerospace)
]
```

### 加工参数优化

```python
# Heat treatment optimization
objectives = [
    "Hardness",          # Maximize
    "Toughness",         # Maximize
    "Residual Stress",   # Minimize
    "Energy Cost",       # Minimize
    "Processing Time"    # Minimize
]
```

### 多功能材料

```python
# Electronic materials
objectives = [
    "Electrical Conductivity",  # Maximize
    "Thermal Conductivity",     # Maximize/Minimize (depends on application)
    "Mechanical Strength",      # Maximize
    "Corrosion Resistance",     # Maximize
    "Manufacturing Cost"        # Minimize
]
```

## 可视化技术

### 二维帕累托前沿

```python
import matplotlib.pyplot as plt

# Plot Pareto front
plt.scatter(strength_values, ductility_values, c='red', label='Pareto Front')
plt.xlabel('Strength (MPa)')
plt.ylabel('Ductility (%)')
plt.title('Strength vs. Ductility Trade-off')
```

### 3D 帕累托前沿

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(strength, ductility, cost, c='red')
ax.set_xlabel('Strength')
ax.set_ylabel('Ductility') 
ax.set_zlabel('Cost')
```

### 平行坐标

对于 >3 个目标，使用平行坐标图：

```python
import pandas as pd
from pandas.plotting import parallel_coordinates

# Create DataFrame with objectives
df = pd.DataFrame({
    'Strength': strength_values,
    'Ductility': ductility_values,
    'Cost': cost_values,
    'Corrosion': corrosion_values,
    'Type': 'Pareto'
})

parallel_coordinates(df, 'Type')
```

## 决策

### 基于偏好的选择

找到帕累托前沿后，根据喜好选择解决方案：

```python
# Weight-based selection
weights = [0.4, 0.3, 0.3]  # [strength, ductility, cost]

# Calculate weighted sum for each Pareto solution
scores = []
for solution in pareto_front:
    score = sum(w * obj for w, obj in zip(weights, solution))
    scores.append(score)

# Select solution with highest score
best_index = np.argmax(scores)
selected_solution = pareto_front[best_index]
```

### 拐点选择

“拐点”代表最佳折衷方案：

```python
# Find knee point (maximum distance from line connecting extremes)
def find_knee_point(pareto_front):
    # Implementation depends on specific algorithm
    # Common approaches: maximum perpendicular distance, angle-based
    pass
```

## 下一步

现在您已经了解了多目标概念：

1. **学习MOBO算法**：{doc}`mobo_algorithms`
2. **尝试 MultiBgolearn**：{doc}`multibgolearn`
3. **举例练习**：{doc}`examples/multi_objective`
4. **探索帕累托优化**：{doc}`pareto_optimization`

```{seealso}
为了更深入的理解：
- Deb, K.“使用进化算法的多目标优化”
- 科埃洛，C.A.C. 《解决多目标问题的进化算法》
- Miettinen, K.“非线性多目标优化”
```
