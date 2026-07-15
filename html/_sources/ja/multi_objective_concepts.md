# Multi-Objective Optimization Concepts

```{note}
本ページは Bgolearn マニュアルの日本語版です。
```

```{note}
This page introduces the fundamental concepts of multi-objective optimization and how they apply to materials design problems.
```

## What is Multi-Objective Optimization?

Multi-objective optimization (MOO) deals with problems where we need to optimize multiple, often conflicting objectives simultaneously. Unlike single-objective optimization that seeks one optimal solution, MOO typically results in a set of trade-off solutions called the **Pareto front**.

```{admonition} Real-World Example
:class: tip
Consider designing a new alloy:
- **Objective 1**: Maximize strength (higher is better)
- **Objective 2**: Maximize ductility (higher is better)  
- **Objective 3**: Minimize cost (lower is better)

These objectives often conflict: stronger alloys may be more brittle (less ductile) and more expensive.
```

## Key Concepts

### Pareto Dominance

A solution **dominates** another if it's better in at least one objective and no worse in all others.

```python
# Example: Two alloy compositions
Alloy_A = [Strength=250, Ductility=15, Cost=100]
Alloy_B = [Strength=240, Ductility=15, Cost=100]

# Alloy A dominates Alloy B because:
# - Strength: 250 > 240 (better)
# - Ductility: 15 = 15 (equal)
# - Cost: 100 = 100 (equal)
```

### Pareto Front

The **Pareto front** (or Pareto frontier) is the set of all non-dominated solutions. These represent the best possible trade-offs between objectives.

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

### Pareto Optimality

A solution is **Pareto optimal** if no other solution dominates it. The goal of multi-objective optimization is to find the Pareto front.

## Mathematical Formulation

### General Multi-Objective Problem

```
minimize/maximize: f(x) = [f₁(x), f₂(x), ..., fₘ(x)]
subject to: g(x) ≤ 0
           h(x) = 0
           x ∈ X
```

Where:
- `f(x)`: Vector of m objective functions
- `g(x)`: Inequality constraints
- `h(x)`: Equality constraints
- `X`: Decision variable space

### Materials Design Example

For alloy optimization:

```
maximize: f₁(x) = Strength(Cu, Mg, Si, T_aging)
maximize: f₂(x) = Ductility(Cu, Mg, Si, T_aging)
minimize: f₃(x) = Cost(Cu, Mg, Si, T_aging)

subject to: Cu + Mg + Si ≤ 10%  (composition constraint)
           150 ≤ T_aging ≤ 200   (temperature constraint)
           Cu, Mg, Si ≥ 0        (non-negativity)
```

## Multi-Objective vs. Single-Objective

```{list-table} Comparison
:header-rows: 1
:name: mo-vs-so

* - Aspect
  - Single-Objective
  - Multi-Objective
* - **Solution**
  - One optimal point
  - Set of trade-off solutions (Pareto front)
* - **Decision Making**
  - Automatic
  - Requires preference information
* - **Complexity**
  - Lower
  - Higher (curse of dimensionality)
* - **Visualization**
  - Easy (1D or 2D)
  - Difficult (high-dimensional)
* - **Algorithms**
  - Well-established
  - More complex, newer field
```

## Challenges in Multi-Objective Optimization

### 1. Conflicting Objectives

Most real-world problems involve trade-offs:

```python
# Materials examples
conflicts = {
    "Strength vs. Ductility": "Stronger materials are often more brittle",
    "Performance vs. Cost": "Better performance usually costs more",
    "Conductivity vs. Stability": "High conductivity may reduce thermal stability",
    "Hardness vs. Toughness": "Harder materials may be less tough"
}
```

### 2. Curse of Dimensionality

As the number of objectives increases, the problem becomes exponentially more complex:

- **2-3 objectives**: Manageable, good visualization
- **4-6 objectives**: Challenging, limited visualization
- **>6 objectives**: Very difficult, many-objective optimization

### 3. Solution Selection

The Pareto front provides multiple solutions, but we often need to select one:

- **A priori**: Define preferences before optimization
- **A posteriori**: Choose from Pareto front after optimization
- **Interactive**: Iteratively refine preferences during optimization

## Multi-Objective Bayesian Optimization (MOBO)

MOBO extends Bayesian optimization to handle multiple objectives by:

1. **Surrogate Models**: Build separate models for each objective
2. **Multi-Objective Acquisition**: Use acquisition functions that consider all objectives
3. **Pareto Front Approximation**: Iteratively improve the Pareto front estimate

### Key Advantages

```{admonition} Why MOBO for Materials?
:class: tip
- **Expensive Experiments**: Materials testing is costly and time-consuming
- **Multiple Properties**: Materials have many important properties
- **Trade-off Understanding**: Need to understand relationships between properties
- **Efficient Exploration**: Find Pareto front with minimal experiments
```

## MOBO Acquisition Functions

### Expected Hypervolume Improvement (EHVI)

EHVI maximizes the expected improvement in hypervolume (volume of dominated objective space):

```
EHVI(x) = E[HV(F ∪ {f(x)}) - HV(F)]
```

Where:
- `F`: Current Pareto front approximation
- `HV`: Hypervolume indicator
- `f(x)`: Objective vector at point x

**Best for**: 2-4 objectives, balanced exploration



### q-Noisy Expected Hypervolume Improvement (qNEHVI)

qNEHVI extends EHVI to handle noisy observations and batch acquisition, making it suitable for real-world scenarios with measurement uncertainty and parallel experiments:

```
qNEHVI(x) = E_y[E[HV(F ∪ {f(x)}) - HV(F)]]
```

Where:
- `y`: Noisy observations
- `HV`: Hypervolume indicator
- `f(x)`: Objective vector at point x

**Best for**: Measurements have significant observation noise



### Probability of Improvement (PI)

Multi-objective PI considers improvement probability for each objective:

```
PI(x) = P(f(x) dominates at least one point in F)
```

**Best for**: Conservative optimization, exploitation-focused

### Upper Confidence Bound (UCB)

Multi-objective UCB balances mean prediction and uncertainty:

```
UCB(x) = μ(x) + β·σ(x)
```

Applied to each objective separately or combined.

**Best for**: Noisy objectives, exploration-focused

## Hypervolume Indicator

The hypervolume is a key quality indicator for Pareto fronts:

```
HV(F) = Volume of space dominated by F
```

### Properties
- **Monotonic**: Larger hypervolume = better Pareto front
- **Pareto Compliant**: Respects Pareto dominance
- **Reference Point Dependent**: Requires a reference point

### Calculation Example

For 2D objectives (maximize both):

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

## Practical Considerations

### Objective Scaling

Different objectives may have different scales:

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

### Reference Point Selection

For hypervolume calculation, choose reference points carefully:

```python
# Good reference point (slightly worse than worst known values)
reference_point = [min_strength - 0.1*range_strength,
                  min_ductility - 0.1*range_ductility,
                  max_cost + 0.1*range_cost]  # Note: max for minimization
```

### Objective Transformation

Convert minimization to maximization:

```python
# Original objectives
strength = 250      # maximize
ductility = 15      # maximize  
cost = 100          # minimize

# Transformed for maximization
objectives = [strength, ductility, -cost]  # Negate cost
```

## Materials Design Applications

### Alloy Composition Optimization

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

### Processing Parameter Optimization

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

### Multi-Functional Materials

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

## Visualization Techniques

### 2D Pareto Front

```python
import matplotlib.pyplot as plt

# Plot Pareto front
plt.scatter(strength_values, ductility_values, c='red', label='Pareto Front')
plt.xlabel('Strength (MPa)')
plt.ylabel('Ductility (%)')
plt.title('Strength vs. Ductility Trade-off')
```

### 3D Pareto Front

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(strength, ductility, cost, c='red')
ax.set_xlabel('Strength')
ax.set_ylabel('Ductility') 
ax.set_zlabel('Cost')
```

### Parallel Coordinates

For >3 objectives, use parallel coordinate plots:

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

## Decision Making

### Preference-Based Selection

After finding the Pareto front, select solutions based on preferences:

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

### Knee Point Selection

The "knee point" represents the best compromise:

```python
# Find knee point (maximum distance from line connecting extremes)
def find_knee_point(pareto_front):
    # Implementation depends on specific algorithm
    # Common approaches: maximum perpendicular distance, angle-based
    pass
```

## Next Steps

Now that you understand multi-objective concepts:

1. **Learn MOBO algorithms**: {doc}`mobo_algorithms`
2. **Try MultiBgolearn**: {doc}`multibgolearn`
3. **Practice with examples**: {doc}`examples/multi_objective`
4. **Explore Pareto optimization**: {doc}`pareto_optimization`

```{seealso}
For deeper understanding:
- Deb, K. "Multi-Objective Optimization using Evolutionary Algorithms"
- Coello, C.A.C. "Evolutionary Algorithms for Solving Multi-Objective Problems"
- Miettinen, K. "Nonlinear Multiobjective Optimization"
```
