# MOBO Algorithms in MultiBgolearn

```{note}
This page provides detailed explanations of the multi-objective Bayesian optimization algorithms implemented in MultiBgolearn.
```

## Overview

MultiBgolearn implements three main multi-objective acquisition functions, each with different strengths and use cases:

1. **Expected Hypervolume Improvement (EHVI)** - Volume-based optimization
2. **q-Noisy Expected Hypervolume Improvement (qNEHVI)** - Volume-based optimization
3. **Probability of Improvement (PI)** - Improvement probability-based
4. **Upper Confidence Bound (UCB)** - Uncertainty-aware exploration

## Expected Hypervolume Improvement (EHVI)

### Theory

EHVI is considered the gold standard for multi-objective Bayesian optimization. It maximizes the expected improvement in hypervolume, which is the volume of objective space dominated by the Pareto front.

```
EHVI(x) = E[HV(F ∪ {f(x)}) - HV(F)]
```

Where:
- `F`: Current Pareto front approximation
- `HV(·)`: Hypervolume indicator
- `f(x)`: Predicted objective vector at point x

### Mathematical Details

For a Gaussian Process surrogate model, the EHVI can be computed analytically for 2D problems and approximated for higher dimensions:

```
EHVI(x) = ∫ max(0, HV_improvement) · p(f(x)) df(x)
```

Where `p(f(x))` is the multivariate Gaussian distribution from the GP predictions.

### Implementation in MultiBgolearn

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

### Advantages

```{admonition} EHVI Strengths
:class: tip
- **Theoretically Sound**: Maximizes a well-defined quality indicator
- **Pareto Compliant**: Respects Pareto dominance relationships
- **Balanced Exploration**: Good trade-off between exploration and exploitation
- **Scalable**: Works well for 2-4 objectives
```

### Limitations

```{admonition} EHVI Limitations
:class: warning
- **Computational Cost**: Expensive for >4 objectives
- **Reference Point Dependent**: Requires careful reference point selection
- **Approximation Needed**: Exact computation only feasible for 2D
```

### Best Use Cases

- **2-4 objectives**: Optimal performance range
- **Balanced exploration**: When you want comprehensive Pareto front
- **Quality focus**: When hypervolume is important metric
- **Materials design**: Alloy optimization, processing parameters

### Example: Alloy Design with EHVI

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
## q-Noisy Expected Hypervolume Improvement (qNEHVI)

### Theory

qNEHVI extends EHVI to handle noisy observations and batch acquisition, making it the state-of-the-art method for real-world multi-objective optimization with measurement uncertainty and parallel experiments.

```
qNEHVI(X_q) = E_y[E_f[HV(P(y ∪ f_q)) - HV(P(y))]]
```


Where:
- `X_q = {x_1, ..., x_q}`: Batch of q candidate points
- `y`: Noisy observations from existing data
- `f_q`: Predicted objectives for candidate batch
- `P(·)`: Pareto front operator
- Inner expectation: Over GP posterior predictions
- Outer expectation: Over observation noise

### Mathematical Details

qNEHVI explicitly models observation noise in the acquisition function:

```
y_i(x) = f_i(x) + ε_i, ε_i ~ N(0, σ²_obs,i)
```

The acquisition function is approximated via nested Monte Carlo sampling:
```
qNEHVI(X_q) ≈ (1/S₁S₂) Σ Σ max(0, HV(P(y^(s₁) ∪ f_q^(s₂))) - HV(P(y^(s₁))))
```



### Example: Alloy Design with qNEHVI

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

## Probability of Improvement (PI)

### Theory

Multi-objective PI calculates the probability that a candidate point will improve upon at least one objective in the current Pareto front.

```
PI(x) = P(f(x) dominates at least one point in F)
```

For Gaussian Process predictions:

```
PI(x) = P(∃ y ∈ F : f(x) ≻ y)
```

Where `≻` denotes Pareto dominance.

### Implementation

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

### Calculation Details

For each candidate point x, PI is computed as:

1. **Generate samples** from GP posterior: `f_samples ~ GP(x)`
2. **Check dominance** for each sample against Pareto front
3. **Calculate probability** as fraction of dominating samples

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

### Advantages

```{admonition} PI Strengths
:class: tip
- **Intuitive**: Easy to understand and interpret
- **Conservative**: Focuses on likely improvements
- **Fast Computation**: Relatively efficient to calculate
- **Robust**: Works well with noisy objectives
```

### Limitations

```{admonition} PI Limitations
:class: warning
- **Exploitation Bias**: May not explore enough
- **Local Optima**: Can get stuck in local regions
- **Magnitude Ignorance**: Doesn't consider improvement size
```

### Best Use Cases

- **Conservative optimization**: When you want reliable improvements
- **Noisy objectives**: When measurements have high uncertainty
- **Limited budget**: When you have few evaluation opportunities
- **Exploitation focus**: When you want to improve known good regions

### Example: Processing Parameter Optimization

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

## Upper Confidence Bound (UCB)

### Theory

Multi-objective UCB balances exploitation (mean prediction) and exploration (uncertainty) by using confidence bounds for each objective.

```
UCB(x) = μ(x) + β·σ(x)
```

Where:
- `μ(x)`: Mean prediction vector
- `σ(x)`: Standard deviation vector  
- `β`: Exploration parameter

### Multi-Objective UCB Variants

#### 1. Component-wise UCB
Apply UCB to each objective separately:

```
UCB_i(x) = μ_i(x) + β·σ_i(x)  for i = 1, ..., m
```

#### 2. Hypervolume-based UCB
Use UCB values to compute hypervolume:

```
UCB_HV(x) = HV([UCB_1(x), UCB_2(x), ..., UCB_m(x)])
```

#### 3. Scalarized UCB
Combine objectives with weights:

```
UCB_scalar(x) = Σ w_i · UCB_i(x)
```

### Implementation

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

### Parameter Selection

The exploration parameter β controls the exploration-exploitation trade-off:

```python
# β selection guidelines
beta_values = {
    "Conservative (exploitation)": 0.5,
    "Balanced": 1.0,
    "Aggressive (exploration)": 2.0,
    "Very aggressive": 3.0
}
```

### Advantages

```{admonition} UCB Strengths
:class: tip
- **Uncertainty Aware**: Explicitly considers prediction uncertainty
- **Tunable**: β parameter allows exploration control
- **Robust**: Works well with noisy data
- **Scalable**: Efficient for many objectives
```

### Limitations

```{admonition} UCB Limitations
:class: warning
- **Parameter Tuning**: Requires β selection
- **Overexploration**: May explore too much with high β
- **Model Dependent**: Performance depends on uncertainty estimates
```

### Best Use Cases

- **Noisy objectives**: When measurements have significant uncertainty
- **Exploration needs**: When you want to explore unknown regions
- **High dimensions**: When you have many objectives (>4)
- **Uncertainty quantification**: When prediction confidence matters

### Example: Multi-Property Ceramic Design

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

## Algorithm Comparison

### Performance Characteristics

```{list-table} Algorithm Comparison
:header-rows: 1
:name: mobo-comparison

* - Algorithm
  - Best For
  - Objectives
  - Exploration
  - Computation
* - **EHVI**
  - Balanced optimization
  - 2-4
  - Moderate
  - High
* - **qNEHVI**
  - Balanced optimization
  - 2-4
  - Moderate
  - Vary High
* - **PI**
  - Conservative improvement
  - 2-6
  - Low
  - Medium
* - **UCB**
  - Uncertain/noisy data
  - 2-8+
  - High
  - Low
```

### Selection Guidelines

#### Choose EHVI when:
- You have 2-4 objectives
- You want comprehensive Pareto front
- Computational resources are available
- Hypervolume is important metric


#### Choose qNEHVI when:
- You have 2-4 objectives
- Measurements have significant observation noise or uncertainty
- Multiple experiments can be conducted in parallel



#### Choose PI when:
- You want reliable improvements
- Objectives are noisy
- Limited evaluation budget
- Conservative approach preferred

#### Choose UCB when:
- You have >4 objectives
- High uncertainty in measurements
- Need to explore unknown regions
- Want tunable exploration

## Advanced Topics

### Constraint Handling

All algorithms can handle constraints through penalty methods:

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

### Batch Optimization

For parallel experiments, modify acquisition functions:

```python
# Batch EHVI (simplified concept)
def batch_EHVI(X_batch, pareto_front, gp_models):
    # Compute joint improvement for entire batch
    # More complex than single-point EHVI
    pass
```

### Dynamic Objectives

Handle changing objectives over time:

```python
# Update objectives based on new requirements
def update_objectives(iteration):
    if iteration < 10:
        return ['strength', 'ductility']
    else:
        return ['strength', 'ductility', 'cost']  # Add cost later
```

## Implementation Tips

### Data Preprocessing

```python
# Normalize objectives to similar scales
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
objectives_normalized = scaler.fit_transform(objectives)
```

### Model Selection

```python
# Model recommendations by algorithm
model_recommendations = {
    'EHVI': 'GaussianProcess',      # Best uncertainty estimates
    'qNEHVI': 'GaussianProcess',      # Best uncertainty estimates
    'PI': 'RandomForest',           # Robust to noise
    'UCB': 'GaussianProcess'        # Good uncertainty quantification
}
```

### Bootstrap Settings

```python
# Bootstrap recommendations by algorithm
bootstrap_settings = {
    'EHVI': 5,   # Moderate bootstrap for balance
    'qNEHVI': 5,   # Moderate bootstrap for balance
    'PI': 8,     # Higher for noise robustness  
    'UCB': 10    # Highest for uncertainty estimation
}
```

## Troubleshooting

### Common Issues

1. **Poor Convergence**
   - Increase bootstrap iterations
   - Try different surrogate models
   - Check objective scaling

2. **Slow Performance**
   - Reduce virtual space size
   - Use simpler models (RandomForest vs GaussianProcess)
   - Decrease bootstrap iterations

3. **Unexpected Recommendations**
   - Verify objective definitions (max vs min)
   - Check data preprocessing
   - Validate constraint handling

### Performance Optimization

```python
# Speed up optimization
optimization_tips = {
    "Reduce virtual space": "Keep <5000 candidates",
    "Simplify models": "Use RandomForest for speed",
    "Parallel bootstrap": "Use multiprocessing",
    "Cache computations": "Store expensive calculations"
}
```

## Next Steps

Now that you understand MOBO algorithms:

1. **Practice with examples**: {doc}`examples/multi_objective`
2. **Learn Pareto analysis**: {doc}`pareto_optimization`
3. **Try real applications**: {doc}`examples/materials_design`
4. **Explore advanced features**: {doc}`examples/advanced_usage`

```{seealso}
For algorithm details:
- Emmerich, M. "Single- and Multi-objective Evolutionary Optimization"
- Jones, D.R. "Efficient Global Optimization of Expensive Black-Box Functions"
- Knowles, J. "ParEGO: A Hybrid Algorithm with On-line Landscape Approximation"
- Samuel, D. "Parallel bayesian optimization of multiple noisy objectives with expected hypervolume improvement"
```
