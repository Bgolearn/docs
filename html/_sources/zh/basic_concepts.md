# 基本概念

```{note}
本页是 Bgolearn 手册的中文版本。
```

```{note}
本页介绍贝叶斯优化的基本概念以及 Bgolearn 如何实现它们。
```

## 什么是贝叶斯优化？

贝叶斯优化是一种用于优化评估成本高昂的函数的强大技术。它在以下情况下特别有用：

- **功能评估成本高昂**（实验、模拟）
- **衍生品不可用**（黑盒函数）
- **测量中存在噪音**
- **几乎无法进行评估**（预算有限）

```{admonition} 核心思想
:class: tip
贝叶斯优化不是随机采样或使用网格搜索，而是构建函数的概率模型，并使用它来智能地决定下一步在哪里采样。
```

## 核心组件

### 1. 代理模型

代理模型使用先前的观察结果来近似昂贵的函数。

**高斯过程 (GP)** 是最常见的选择：
- 提供**平均预测**和**不确定性估计**
- 自然地处理观察中的噪音
- 灵活且适合解决许多问题

```python
# Example: Fitting a GP model in Bgolearn
from Bgolearn import BGOsampling

optimizer = BGOsampling.Bgolearn()
model = optimizer.fit(
    data_matrix=X_train,
    Measured_response=y_train,
    virtual_samples=X_candidates
)

# Get predictions with uncertainty
mean_pred = model.virtual_samples_mean
std_pred = model.virtual_samples_std
```

### 2. 采集功能

采集函数通过平衡来决定下一步采样的位置：
- **利用**：模型预测良好值的样本
- **探索**：不确定性较高的样本

Bgolearn中常用采集函数：

```{list-table} Acquisition Functions
:header-rows: 1
:name: acquisition-functions-table

* - 功能
  - 描述
  - 最适合
* - **EI**（预期改进）
  - 预计较当前最佳状态有所改善
  - 通用、平衡的探索/利用
* - **UCB**（置信上限）
  - 充满信心的乐观估计
  - 嘈杂的功能，专注于探索
* - **PI**（改进的概率）
  - 改善当前最佳状态的概率
  - 保守，注重剥削
* - **PES**（预测熵搜索）
  - 信息论方法
  - 功能复杂，预算有限
```

### 3. 优化循环

贝叶斯优化过程遵循以下迭代循环：

```
1. Initial Data
   ↓
2. Fit Surrogate Model
   ↓
3. Optimize Acquisition Function
   ↓
4. Evaluate at New Point
   ↓
5. Update Dataset
   ↓
6. Stopping Criterion?
   ├─ No → Go back to step 2
   └─ Yes → Return Best Solution
```

## 数学基础

### 高斯过程

高斯过程定义为：
- **均值函数**：$m(x) = \mathbb{E}[f(x)]$
- **协方差函数**：$k(x, x') = \text{Cov}[f(x), f(x')]$

对于任何有限的点集，函数值遵循多元高斯分布：

$$f(x_1), \ldots, f(x_n) \sim \mathcal{N}(\mu, K)$$

其中 $\mu_i = m(x_i)$ 和 $K_{ij} = k(x_i, x_j)$。

### 预期改善

预期改进获取函数为：

$$\text{EI}(x) = \mathbb{E}[\max(f(x) - f^*, 0)]$$

其中 $f^*$ 是当前的最佳观测值。

对于均值 $\mu(x)$ 和方差 $\sigma^2(x)$ 的 GP 后验：

$$\text{EI}(x) = (\mu(x) - f^*)\Phi(Z) + \sigma(x)\phi(Z)$$

其中 $Z = \frac{\mu(x) - f^*}{\sigma(x)}$、$\Phi$ 是 CDF，$\phi$ 是标准正态分布的 PDF。

## 实际考虑

### 何时使用贝叶斯优化

✅ **适合：**
- 昂贵的功能评估（每次评估 >1 秒）
- 连续或混合变量空间
- 嘈杂的观察
- 评估预算有限（10-1000 次评估）
- 无导数的黑盒函数

❌ **不适合：**
- 非常便宜的函数（使用基于梯度的方法）
- 非常高的维度（>20 个变量）
- 离散组合问题
- 结构已知的函数

### 选择采集功能

```{admonition} 快速指南
:class: tip

- **从 EI 开始**：良好的通用选择
- **使用 UCB 处理噪声函数**：更好的探索
- **尝试利用 PI 进行开发**：当你想保守一点时
- **考虑将 PES 用于复杂功能**：信息论方法
```

### 处理约束

Bgolearn支持多种约束类型：

1. **框约束**：变量的简单界限
2. **线性约束**：线性等式/不等式约束
3. **非线性约束**：一般约束函数
4. **分类变量**：离散选择

```python
# Example: Box constraints
bounds = {
    'temperature': (100, 500),  # Temperature range
    'pressure': (1, 10),        # Pressure range
    'composition': (0, 1)       # Composition fraction
}
```

## 下一步

现在您已经了解了基础知识：

1. **了解采集函数**：{doc}`acquisition_functions`


```{seealso}
有关更深入的数学背景，请参阅：
- 高斯过程的 `Mockus et al., The application of Bayesian methods for seeking the extremum`
- `Zhan et al.,Expected improvement for expensive optimization: a review`为EI原论文

```
