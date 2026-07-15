# Bgolearn 

```{note}
Welcome to the documentation of **Bgolearn** : *a unified Bayesian optimization framework for accelerating materials discovery*. This document is written and produced by **Dr. [Bin Cao](https://bin-cao.github.io/)** to help new learners master the basics of Bayesian Optimization and use **Bgolearn** to solve real-world optimization problems.

For beginners, **[Bgolearn Playground](https://bin-cao.github.io/Bgolearn/)** provides a game-like way to understand the full Bayesian optimization workflow.
```

```{admonition} What is Bgolearn?
:class: tip
Bgolearn is a Python package developed by **Dr. [Bin Cao](https://bin-cao.github.io/) at Hong Kong University of Science and Technology (Guangzhou)** that implements state-of-the-art Bayesian optimization algorithms for both single-objective and multi-objective optimization. It's particularly powerful for materials discovery, where experiments are costly and time-consuming. 


**Key Features:**
- Single-objective optimization with multiple acquisition functions
- Multi-objective optimization via MultiBgolearn
- Materials-focused design and applications
- Flexible surrogate model selection
- Bootstrap uncertainty quantification
```

```{admonition} What makes Bgolearn special?
:class: tip
- **Materials-focused**: Built specifically for materials science workflows
- **Parallel experiments**: Batch optimization for efficient resource utilization
- **Rich visualizations**: Interactive plots and optimization dashboards
- **Robust & reliable**: Comprehensive error handling and data validation
- **Easy to use**: Simple API with sensible defaults
```

```{admonition} About
:class: tip

The *Bgolearn* project is supported by the **Shanghai Artificial Intelligence Open Source Award Project Support Plan (上海市人工智能开源奖励项目支持计划, 2025)**, with a funding of 500,000 RMB. 

An updated survey of *Bgolearn* has been published on **[npj Computational Materials, 2026](
https://doi.org/10.48550/arXiv.2601.06820)**. We express our deep gratitude for the guidance and contributions of **Prof. Tong-Yi Zhang** (Shanghai University, HKUST(GZ)), **Prof. Jun Wang** (University College London), **Prof. Turab Lookman** (Xi'an Jiaotong University), **Prof. Dezhen Xue** (Xi'an Jiaotong University), **Prof. Jie Xiong** (Shanghai University), and **Prof. Jian Hui** (Suzhou Laboratory).



### Key Features

::::{grid} 2
:::{grid-item-card} Materials Science Focus
:class-header: bg-light
Specialized workflows for composition optimization, processing parameter tuning, and multi-objective materials design.
:::

:::{grid-item-card} Multiple Acquisition Functions
:class-header: bg-light
EI, UCB, PI, PES, KG, etc. for different optimization strategies and experimental constraints.
:::

:::{grid-item-card} Batch Optimization
:class-header: bg-light
Select multiple experiments for parallel execution, dramatically reducing optimization time and cost.
:::

:::{grid-item-card} Advanced Visualization
:class-header: bg-light
Interactive plots, optimization dashboards, and materials-specific visualizations for better insights.
:::
::::

### Quick Start Example

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

## Installation

````{tab-set}
```{tab-item} Basic Installation
```bash
pip install bgolearn
```

```{tab-item} Development Installation
```bash
git clone https://github.com/Bin-Cao/Bgolearn.git
cd Bgolearn
pip install -e .
```

```{tab-item} With Optional Dependencies
```bash
# Complete installation with all features
pip install bgolearn[all]

# Or install specific components
pip install bgolearn plotly seaborn  # For advanced visualization
pip install bgolearn pymatgen matminer  # For materials science
```
````

```{admonition} Verify Installation
:class: note
```python
from Bgolearn import BGOsampling
print("Bgolearn imported successfully!")

# Test basic functionality
opt = BGOsampling.Bgolearn()
print("Bgolearn optimizer initialized successfully!")
```

## Documentation Structure

This documentation is organized into several main sections:

::::{grid} 1 1 2 4
:::{grid-item-card} Getting Started
:link: getting_started
:link-type: doc
:class-header: bg-primary text-white

Installation, basic concepts, and your first optimization
+++
Perfect for newcomers to Bayesian optimization
:::

:::{grid-item-card} BgoFace GUI
:link: bgoface
:link-type: doc
:class-header: bg-warning text-dark

Graphical user interface for visual optimization workflows
+++
No coding required - perfect for materials scientists
:::

:::{grid-item-card} Core Documentation
:link: api_reference
:link-type: doc
:class-header: bg-info text-white

API reference, acquisition functions, and optimization strategies
+++
Comprehensive guides for all features
:::

:::{grid-item-card} Applications
:link: materials_discovery
:link-type: doc
:class-header: bg-success text-white

Materials discovery workflows and specialized applications
+++
Real-world examples and best practices
:::
::::

### Detailed Navigation

```{tableofcontents}
```

## Learning Paths

Choose your learning path based on your background and goals:

````{tab-set}
```{tab-item} 🔰 Beginner Path
**New to Bayesian Optimization?**

1. {doc}`getting_started` - Installation and basic concepts
2. {doc}`first_optimization` - Your first optimization tutorial
3. {doc}`acquisition_functions` - Understanding acquisition functions
4. {doc}`examples/single_objective` - Single-objective examples
```

```{tab-item} 🔬 Materials Scientist Path
**Focused on materials discovery?**

1. {doc}`materials_discovery` - Materials discovery overview
2. {doc}`examples/single_objective` - Alloy composition optimization
3. {doc}`multibgolearn` - Multi-objective optimization
4. {doc}`examples/multi_objective` - Multi-property design examples
```

```{tab-item} ⚡ Advanced User Path
**Ready for advanced features?**

1. {doc}`optimization_strategies` - Advanced optimization strategies
2. {doc}`surrogate_models` - Different surrogate models
3. {doc}`bgoface` - GUI interface for visual workflows
4. {doc}`multibgolearn` - Multi-objective optimization
```
````

## Community & Support

```{admonition} Get Help & Connect
:class: tip

- **GitHub Discussions**: Ask questions and share experiences
- **Issues**: Report bugs and request features
- **Email**: binjacobcao@gmail.com
- **Documentation**: You're reading it!
```

::::{grid} 2
:::{grid-item-card} 🔗 Links
- [GitHub Repository](https://github.com/Bin-Cao/Bgolearn)
- [PyPI Package](https://pypi.org/project/bgolearn/)
- [Documentation](https://bgolearn.readthedocs.io/)
:::

:::{grid-item-card} Contributing
- [Contributing Guide](contributing)
- [Code of Conduct](https://github.com/Bin-Cao/Bgolearn/blob/main/CODE_OF_CONDUCT.md)
- [Development Setup](contributing.md#development-setup)
:::
::::

## Citation

If you use Bgolearn in your research, please cite our work:

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

## License

```{admonition} MIT License
:class: note
Bgolearn is released under the MIT License, making it free for both academic and commercial use.
See the [full license](https://github.com/Bin-Cao/Bgolearn/blob/main/LICENSE) for details.
```

---

**Ready to accelerate your research?** {doc}`Start here <getting_started>` or explore our {doc}`examples <examples/index>` to see Bgolearn in action! 
