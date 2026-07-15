# MultiBgolearn: Multi-Objective Bayesian Global Optimization

```{note}
本ページは Bgolearn マニュアルの日本語版です。
```

```{note}
MultiBgolearn extends Bgolearn to handle multi-objective optimization problems, enabling simultaneous optimization of multiple competing objectives in materials design.
```

## Overview

MultiBgolearn is a Python package designed for multi-objective Bayesian global optimization (MOBO), specifically tailored for materials design. It extends the functionalities of the Bgolearn package by enabling the simultaneous optimization of multiple material properties, making it highly suitable for real-world applications where trade-offs between competing objectives are common.

```{admonition} Why Multi-Objective Optimization?
:class: tip
In materials design, we often need to optimize multiple properties simultaneously:
- **Strength vs. Ductility**: Stronger materials are often more brittle
- **Performance vs. Cost**: Better performance usually comes at higher cost
- **Conductivity vs. Thermal Stability**: High conductivity materials may be thermally unstable
- **Corrosion Resistance vs. Mechanical Properties**: Anti-corrosion treatments may affect strength

MultiBgolearn helps find the optimal trade-offs between these competing objectives.
```

## Key Features

### Multiple MOBO Algorithms
- **Expected Hypervolume Improvement (EHVI)**: Maximizes the volume of objective space
- **q-Noisy Expected Hypervolume Improvement (qNEHVI)**:qNEHVI extends EHVI to handle noisy observations and batch acquisition
- **Probability of Improvement (PI)**: Focuses on improvement probability
- **Upper Confidence Bound (UCB)**: Balances exploration and exploitation

### Materials-Focused Design
- Simultaneous optimization of multiple material properties
- Flexible objective handling (maximize/minimize)
- Bootstrap uncertainty quantification
- Automatic model selection

### Flexible Surrogate Models
- RandomForest
- GradientBoosting
- Support Vector Regression (SVR)
- Gaussian Process
- Automatic model selection

## Installation

Install MultiBgolearn using pip:

```bash
pip install MultiBgolearn
```

Or install both packages together:

```bash
pip install Bgolearn MultiBgolearn
```

## Quick Start

Here's a simple example of using MultiBgolearn for materials optimization:

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

## Data Format

### Dataset Format
Your dataset should be a CSV file with the following structure:

```csv
feature1,feature2,feature3,objective1,objective2,objective3
1.2,3.4,5.6,100.5,0.85,7.2
2.1,4.3,6.5,95.2,0.92,6.8
...
```

- **Features**: Input variables (composition, processing conditions, etc.)
- **Objectives**: Target properties to optimize (strength, ductility, cost, etc.)

### Virtual Space Format
The virtual space contains candidate points for optimization:

```csv
feature1,feature2,feature3
1.5,3.2,5.8
2.3,4.1,6.2
...
```

## API Reference

### Main Function: `bgo.fit()`

```python
bgo.fit(dataset_path, VS_path, object_num, max_search=True, 
        method='EHVI', assign_model=False, bootstrap=5)
```

#### Parameters

```{list-table} Parameters
:header-rows: 1
:name: multibgo-parameters

* - Parameter
  - Type
  - Default
  - Description
* - `dataset_path`
  - str
  - Required
  - Path to dataset CSV file
* - `VS_path`
  - str
  - Required
  - Path to virtual space CSV file
* - `object_num`
  - int
  - Required
  - Number of objectives to optimize
* - `max_search`
  - bool
  - True
  - Whether to maximize (True) or minimize (False)
* - `method`
  - str
  - 'EHVI'
  - Optimization method: 'EHVI', 'qNEHVI', 'PI', 'UCB'
* - `assign_model`
  - str/bool
  - False
  - Surrogate model or False for auto-selection
* - `bootstrap`
  - int
  - 5
  - Number of bootstrap iterations
```

#### Returns

```{list-table} Return Values
:header-rows: 1
:name: multibgo-returns

* - Variable
  - Type
  - Description
* - `VS_recommended`
  - array
  - Recommended data point from virtual space
* - `improvements`
  - array
  - Calculated improvements for each objective
* - `index`
  - int
  - Index of recommended point in virtual space
```

## Optimization Methods

### Expected Hypervolume Improvement (EHVI)

EHVI focuses on maximizing the volume of the objective space dominated by the solutions. It's particularly effective for problems with 2-4 objectives.

```python
# Use EHVI for balanced multi-objective optimization
VS_recommended, improvements, index = bgo.fit(
    dataset_path, VS_path, object_num=3,
    method='EHVI',
    assign_model='GaussianProcess'
)
```

**Best for:**
- 2-4 objectives
- Balanced exploration of Pareto front
- When you want to maximize dominated volume


### q-Noisy Expected Hypervolume Improvement (qNEHVI)

qNEHVI extends EHVI to handle noisy observations and batch acquisition, making it suitable for real-world scenarios with measurement uncertainty and parallel experiment.

```python
# Use qNEHVI for balanced multi-objective optimization
VS_recommended, improvements, index = bgo.fit(
    dataset_path, VS_path, object_num=3,
    method='qNEHVI',
    batch_size=3,  # Select 3 points simultaneously
    assign_model='GaussianProcess'
)
```

**Best for:**
- Measurements have significant observation noise or uncertainty
- Multiple experiments can be conducted in parallel
- Known or estimable observation noise levels



### Probability of Improvement (PI)

PI selects points with the highest probability of improving over the best known solution.

```python
# Use PI for improvement-focused optimization
VS_recommended, improvements, index = bgo.fit(
    dataset_path, VS_path, object_num=2,
    method='PI',
    assign_model='RandomForest'
)
```

**Best for:**
- Conservative optimization
- When improvement probability is important
- Exploitation-focused search

### Upper Confidence Bound (UCB)

UCB explores points with the highest upper confidence bound, balancing exploration and exploitation.

```python
# Use UCB for exploration-exploitation balance
VS_recommended, improvements, index = bgo.fit(
    dataset_path, VS_path, object_num=3,
    method='UCB',
    assign_model='GradientBoosting'
)
```

**Best for:**
- Noisy objectives
- When uncertainty matters
- Balanced exploration-exploitation

## Surrogate Models

### Automatic Model Selection

When `assign_model=False`, MultiBgolearn automatically selects the best model:

```python
# Automatic model selection
VS_recommended, improvements, index = bgo.fit(
    dataset_path, VS_path, object_num=3,
    assign_model=False  # Auto-select best model
)
```

### Manual Model Selection

You can specify the surrogate model explicitly:

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

## Bootstrap Uncertainty Quantification

MultiBgolearn uses bootstrap sampling to quantify uncertainty in predictions:

```python
# Increase bootstrap iterations for better uncertainty estimation
VS_recommended, improvements, index = bgo.fit(
    dataset_path, VS_path, object_num=3,
    bootstrap=10  # More iterations = better uncertainty estimation
)
```

**Bootstrap Benefits:**
- Robust uncertainty quantification
- Better handling of noisy data
- Improved model reliability
- More confident recommendations

## Practical Example: Alloy Design

Let's optimize a three-objective alloy design problem:

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

## Best Practices

### 1. Data Preparation
- Ensure sufficient training data (>10 samples per objective)
- Normalize objectives if they have different scales
- Handle missing values appropriately

### 2. Method Selection
- Use **EHVI** for 2-4 objectives with balanced exploration
- Use **qNEHVI** for measurements have significant observation noise or uncertainty
- Use **PI** for conservative, improvement-focused search
- Use **UCB** for noisy objectives or exploration needs

### 3. Model Selection
- Start with automatic model selection
- Use **GaussianProcess** for smooth, continuous objectives
- Use **RandomForest** for discrete or categorical features
- Use **GradientBoosting** for complex, nonlinear relationships

### 4. Bootstrap Settings
- Use 5-10 bootstrap iterations for most problems
- Increase to 20+ for very noisy data
- Balance computation time vs. uncertainty quality

## Troubleshooting

### Common Issues

1. **Convergence Problems**
   - Increase bootstrap iterations
   - Try different surrogate models
   - Check data quality and scaling

2. **Poor Recommendations**
   - Ensure sufficient training data
   - Verify objective definitions (max vs. min)
   - Consider data preprocessing

3. **Computational Issues**
   - Reduce bootstrap iterations
   - Use simpler surrogate models
   - Reduce virtual space size

### Performance Tips

- **Data Size**: Keep virtual space manageable (<10,000 points)
- **Objectives**: EHVI works best with 2-4 objectives
- **Features**: Normalize features to similar scales
- **Iterations**: Start with fewer bootstrap iterations for testing

## Next Steps

- Learn about {doc}`multi_objective_concepts` for theoretical background
- Explore {doc}`mobo_algorithms` for algorithm details
- Try {doc}`examples/multi_objective` for hands-on practice
- Understand {doc}`pareto_optimization` for trade-off analysis

```{seealso}
For more information:
- [MultiBgolearn GitHub Repository](https://github.com/Bin-Cao/MultiBgolearn)
- [Research Papers](https://bgolearn.netlify.app/)
- [Materials Design Examples](examples/materials_design.md)
```
