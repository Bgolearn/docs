# Bgolearn 

```{note}
本页是 Bgolearn 手册的中文版本。
```

```{note}
欢迎来到 **Bgolearn** 的文档：*用于加速材料发现的统一贝叶斯优化框架*。本文件由[曹斌博士](https://bin-cao.github.io/)撰写和制作，帮助新学习者掌握贝叶斯优化的基础知识，并使用 **Bgolearn** 解决现实世界的优化问题。

对于初学者，可以通过 **[Bgolearn Playground](https://bin-cao.github.io/Bgolearn/)** 以游戏的方式了解贝叶斯优化的全流程。
```

```{admonition} Bgolearn 是什么？
:class: tip
Bgolearn 是由 **香港科技大学（广州）的 [曹斌博士](https://bin-cao.github.io/)** 开发的 Python 软件包，为单目标和多目标优化实现了先进的贝叶斯优化算法。它对于材料发现尤其强大，因为材料发现中的实验成本高昂且耗时。


**主要特点：**
- 具有多个采集功能的单目标优化
- 通过 MultiBgolearn 进行多目标优化
- 以材料为中心的设计和应用
- 灵活的代理模型选择
- Bootstrap 不确定性量化
```

```{admonition} Bgolearn 有什么特别之处？
:class: tip
- **以材料为中心**：专为材料科学工作流程而构建
- **并行实验**：批量优化以实现高效的资源利用
- **丰富的可视化**：交互式绘图和优化仪表板
- **稳健可靠**：全面的错误处理和数据验证
- **易于使用**：具有合理默认值的简单 API
```

```{admonition} 关于
:class: tip

*Bgolearn* 项目获得 **上海市人工智能开源奖励项目支持计划（2025）** 支持，资助金额为 50 万元人民币。

*Bgolearn* 的最新论文已发表在 **[npj Computational Materials, 2026](https://arxiv.org/abs/2601.06820)**。我们衷心感谢 **张统一教授**（上海大学、香港科技大学（广州））、**汪军教授**（伦敦大学学院）、**Turab Lookman 教授**（西安交通大学）、**薛祯教授**（西安交通大学）、**熊杰教授**（上海大学）和 **惠健教授**（苏州实验室）的指导和贡献。



### 主要特点

::::{grid} 2
:::{grid-item-card} 材料科学焦点
:class-header: bg-light
用于成分优化、工艺参数调整和多目标材料设计的专业工作流程。
:::

:::{grid-item-card} 多种采集功能
:class-header: bg-light
针对不同优化策略和实验约束的EI、UCB、PI、PES、KG等。
:::

:::{grid-item-card} 批量优化
:class-header: bg-light
选择多个实验并行执行，显着减少优化时间和成本。
:::

:::{grid-item-card} 高级可视化
:class-header: bg-light
交互式绘图、优化仪表板和特定于材料的可视化可提供更好的见解。
:::
::::

### 快速入门示例

```{code-block} python
:linenos:
:emphasize-lines: 8,15,18

from Bgolearn import BGOsampling
import pandas as pd
import numpy as np

# Load your materials data
X_train = pd.DataFrame(np.random.randn(20, 3), columns=['Temperature', 'Pressure', 'Composition'])
y_train = pd.Series(np.random.randn(20))  # Target property (e.g., strength)
X_candidates = pd.DataFrame(np.random.randn(100, 3), columns=['Temperature', 'Pressure', 'Composition'])

# Initialize and fit Bgolearn
optimizer = BGOsampling.Bgolearn()
model = optimizer.fit(
    data_matrix=X_train,  # Pass DataFrame directly
    Measured_response=y_train,  # Pass Series directly
    virtual_samples=X_candidates,  # Pass DataFrame directly
    opt_num=1
)

# Get recommendation using Expected Improvement
ei_values, recommended_points = model.EI()

# The recommended point(s)
next_experiment = recommended_points[0]  # First (best) recommendation
print(f"Next recommended experiment: {next_experiment}")

# Get prediction for all virtual samples
predicted_values = model.virtual_samples_mean
print(f"Number of predictions: {len(predicted_values)}")

# Basic visualization using matplotlib
import matplotlib.pyplot as plt

# Plot EI values
plt.figure(figsize=(10, 6))
plt.plot(ei_values)
plt.title('Expected Improvement Values')
plt.xlabel('Candidate Index')
plt.ylabel('EI Value')
plt.show()
```

## 安装

````{tab-set}
```{tab-item} 基础安装
```bash
pip install bgolearn
```

```{tab-item} 开发版安装
```bash
git clone https://github.com/Bin-Cao/Bgolearn.git
cd Bgolearn
pip install -e .
```

```{tab-item} 安装可选依赖
```bash
# Complete installation with all features
pip install bgolearn[all]

# Or install specific components
pip install bgolearn plotly seaborn  # For advanced visualization
pip install bgolearn pymatgen matminer  # For materials science
```
````

```{admonition} 验证安装
:class: note
```python
from Bgolearn import BGOsampling
print("Bgolearn imported successfully!")

# Test basic functionality
opt = BGOsampling.Bgolearn()
print("Bgolearn optimizer initialized successfully!")
```

## 文档结构

本文档分为几个主要部分：

::::{grid} 1 1 2 4
:::{grid-item-card} 入门
:link: getting_started
:link-type: doc
:class-header: bg-primary text-white

安装、基本概念和您的第一次优化
+++
非常适合贝叶斯优化新手
:::

:::{grid-item-card} BgoFace 图形界面
:link: bgoface
:link-type: doc
:class-header: bg-warning text-dark

用于视觉优化工作流程的图形用户界面
+++
无需编码 - 非常适合材料科学家
:::

:::{grid-item-card} 核心文档
:link: api_reference
:link-type: doc
:class-header: bg-info text-white

API参考、获取函数及优化策略
+++
所有功能的综合指南
:::

:::{grid-item-card} 应用领域
:link: materials_discovery
:link-type: doc
:class-header: bg-success text-white

材料发现工作流程和专业应用
+++
现实世界的例子和最佳实践
:::
::::

### 详细导航

```{tableofcontents}
```

## 学习路径

根据您的背景和目标选择您的学习路径：

````{tab-set}
```{tab-item} 🔰 Beginner Path
**贝叶斯优化新手？**

1. {doc}`getting_started` - 安装和基本概念
2. {doc}`first_optimization` - 您的第一个优化教程
3. {doc}`acquisition_functions` - 了解采集功能
4. {doc}`examples/single_objective` - 单目标示例
```

```{tab-item} 🔬 Materials Scientist Path
**专注于材料发现？**

1. {doc}`materials_discovery` - 材料发现概述
2. {doc}`examples/single_objective` - 合金成分优化
3. {doc}`multibgolearn` - 多目标优化
4. {doc}`examples/multi_objective` - 多属性设计示例
```

```{tab-item} ⚡ Advanced User Path
**准备好使用高级功能了吗？**

1. {doc}`optimization_strategies` - 高级优化策略
2. {doc}`surrogate_models` - 不同的代理模型
3. {doc}`bgoface` - 用于可视化工作流程的 GUI 界面
4. {doc}`multibgolearn` - 多目标优化
```
````

## 社区与支持

```{admonition} 获取帮助与联系
:class: tip

- **GitHub Discussions**：提问并分享经验
- **Issues**：报告错误并提出功能请求
- **Email**：binjacobcao@gmail.com
- **Documentation**：您正在阅读的就是文档！
```

::::{grid} 2
:::{grid-item-card} 🔗 链接
- [GitHub 仓库](https://github.com/Bin-Cao/Bgolearn)
- [PyPI 包](https://pypi.org/project/bgolearn/)
- [文档](https://bgolearn.readthedocs.io/)
:::

:::{grid-item-card} 贡献
- [贡献指南](contributing)
- [行为准则](https://github.com/Bin-Cao/Bgolearn/blob/main/CODE_OF_CONDUCT.md)
- [开发环境设置](contributing.md#development-setup)
:::
::::

## 引用

如果您在研究中使用 Bgolearn，请引用我们的工作：

```{code-block} bibtex
@article{Cao2026Bgolearn,
  author    = {Bin Cao and Jie Xiong and Jiaxuan Ma and Yuan Tian and Yirui Hu and Mengwei He and Longhan Zhang and Jiayu Wang and Jian Hui and Li Liu and Dezhen Xue and Turab Lookman and Jun Wang and Tong-Yi Zhang},
  title     = {Bgolearn: a unified Bayesian optimization framework for accelerating materials discovery},
  journal   = {npj Computational Materials},
  year      = {2026},
  doi       = {10.1038/s41524-026-02226-3},
  issn      = {2057-3960},
  publisher = {Springer Nature},
  url       = {https://doi.org/10.1038/s41524-026-02226-3}
}

```

## 许可证

```{admonition} MIT 许可证
:class: note
Bgolearn 基于 MIT 许可证发布，可免费用于学术和商业用途。
详情请参见[完整许可证](https://github.com/Bin-Cao/Bgolearn/blob/main/LICENSE)。
```

---

**准备好加速您的研究了吗？** {doc}`从这里开始 <getting_started>`，或浏览我们的 {doc}`示例 <examples/index>`，了解 Bgolearn 的实际用法！
