# Pareto Optimization and Trade-off Analysis

```{note}
本ページは Bgolearn マニュアルの日本語版です。
```

```{note}
This page covers Pareto front analysis, trade-off understanding, and decision-making strategies for multi-objective materials design.
```

## Understanding Pareto Fronts

The Pareto front represents the set of optimal trade-offs between competing objectives. In materials design, this helps us understand the fundamental limits and relationships between material properties.

```{admonition} Materials Example
:class: tip
For a structural alloy, the Pareto front might show:
- **High strength, low ductility** (brittle but strong)
- **Medium strength, medium ductility** (balanced properties)  
- **Low strength, high ductility** (tough but weak)

Each point represents an optimal trade-off - you can't improve one property without sacrificing another.
```

## Pareto Front Analysis

### Identifying the Pareto Front

After running MultiBgolearn optimization, analyze the results to identify Pareto optimal solutions:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MultiBgolearn import bgo

# First, create sample data files if they don't exist
print("Creating sample data files...")

# Create sample alloy dataset
alloy_data = pd.DataFrame({
    'Cu': [2.0, 3.5, 1.8, 4.2, 2.8, 3.2, 2.5, 3.8, 2.2, 3.6, 1.9, 4.0, 3.1, 2.7, 3.4],
    'Mg': [1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.9, 1.0, 1.4, 0.7, 1.3, 0.8, 1.0, 1.2, 0.9],
    'Si': [0.5, 0.7, 0.3, 0.8, 0.6, 0.4, 0.9, 0.5, 0.7, 0.6, 0.4, 0.8, 0.6, 0.5, 0.7],
    'Strength': [250, 280, 240, 290, 265, 275, 255, 285, 245, 275, 260, 295, 270, 258, 282],
    'Ductility': [15, 12, 18, 10, 14, 13, 16, 11, 17, 12, 15, 9, 13, 16, 11],
    'Neg_Cost': [-8.5, -9.2, -7.8, -10.1, -8.9, -9.0, -8.3, -9.5, -8.1, -9.3, -8.7, -10.0, -8.8, -8.4, -9.1]
})
alloy_data.to_csv('alloy_dataset.csv', index=False)

# Create virtual space for optimization
np.random.seed(42)
n_candidates = 1000
virtual_space = pd.DataFrame({
    'Cu': np.random.uniform(1.5, 4.5, n_candidates),
    'Mg': np.random.uniform(0.5, 1.5, n_candidates),
    'Si': np.random.uniform(0.3, 1.0, n_candidates)
})
virtual_space.to_csv('virtual_space.csv', index=False)

print(f" Created alloy_dataset.csv with {len(alloy_data)} samples")
print(f" Created virtual_space.csv with {len(virtual_space)} candidates")

# Run multi-objective optimization
VS_recommended, improvements, index = bgo.fit(
    'alloy_dataset.csv',
    'virtual_space.csv',
    object_num=3,
    method='EHVI',
    bootstrap=10
)

# Load your original training dataset
training_data = pd.read_csv('alloy_dataset.csv')

# Create a complete dataset by combining training data with recommended points
# Note: You would need to experimentally measure the objectives for VS_recommended
# For demonstration, we'll use the training data for Pareto analysis
all_data = training_data.copy()

def find_pareto_front(objectives):
    """
    Find Pareto optimal points from objective values.
    
    Parameters:
    objectives: array-like, shape (n_samples, n_objectives)
                Objective values (assuming maximization)
    
    Returns:
    pareto_indices: array of indices of Pareto optimal points
    """
    n_points = objectives.shape[0]
    pareto_indices = []
    
    for i in range(n_points):
        is_pareto = True
        for j in range(n_points):
            if i != j:
                # Check if point j dominates point i
                if all(objectives[j] >= objectives[i]) and any(objectives[j] > objectives[i]):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    
    return np.array(pareto_indices)

# Extract objectives (assuming last 3 columns)
objectives = all_data[['Strength', 'Ductility', 'Neg_Cost']].values
pareto_indices = find_pareto_front(objectives)
pareto_front = objectives[pareto_indices]

print(f"Found {len(pareto_indices)} Pareto optimal solutions")
```

### Combining Training Data with New Experiments

After running optimization and conducting new experiments, combine the results:

```python
# After conducting experiments on recommended points
# Create new experimental data (example)
new_experiments = pd.DataFrame({
    'Cu': [VS_recommended[0]],  # Recommended composition
    'Mg': [VS_recommended[1]],
    'Si': [VS_recommended[2]],
    'Strength': [295],  # Measured values from experiments
    'Ductility': [13],
    'Neg_Cost': [-8.5]
})

# Combine training data with new experimental results
complete_dataset = pd.concat([training_data, new_experiments], ignore_index=True)

# Now perform Pareto analysis on the complete dataset
complete_objectives = complete_dataset[['Strength', 'Ductility', 'Neg_Cost']].values
complete_pareto_indices = find_pareto_front(complete_objectives)
complete_pareto_front = complete_objectives[complete_pareto_indices]

print(f"Updated Pareto front with {len(complete_pareto_indices)} optimal solutions")

# Save the complete dataset for future use
complete_dataset.to_csv('updated_dataset.csv', index=False)
```

### Visualizing Pareto Fronts

#### 2D Pareto Front

```python
# Plot 2D Pareto front
plt.figure(figsize=(10, 8))

# Plot all points
plt.scatter(objectives[:, 0], objectives[:, 1], 
           alpha=0.6, s=50, c='lightblue', label='All solutions')

# Highlight Pareto front
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], 
           c='red', s=100, marker='*', 
           edgecolors='black', linewidth=1, label='Pareto front')

# Connect Pareto points
pareto_sorted = pareto_front[np.argsort(pareto_front[:, 0])]
plt.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 
         'r--', alpha=0.7, linewidth=2)

plt.xlabel('Strength (MPa)')
plt.ylabel('Ductility (%)')
plt.title('Strength vs. Ductility Pareto Front')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### 3D Pareto Front

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot all points
ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
          alpha=0.6, s=30, c='lightblue', label='All solutions')

# Highlight Pareto front
ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
          c='red', s=100, marker='*', 
          edgecolors='black', linewidth=1, label='Pareto front')

ax.set_xlabel('Strength (MPa)')
ax.set_ylabel('Ductility (%)')
ax.set_zlabel('Negative Cost ($/kg)')
ax.set_title('3D Pareto Front: Strength vs. Ductility vs. Cost')
ax.legend()
plt.show()
```


## Trade-off Analysis

### Correlation Analysis

Understand relationships between objectives:

```python
import seaborn as sns

# Calculate correlation matrix
correlation_matrix = pd.DataFrame(pareto_front, 
                                 columns=['Strength', 'Ductility', 'Neg_Cost']).corr()

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
           square=True, fmt='.3f')
plt.title('Objective Correlations in Pareto Front')
plt.tight_layout()
plt.show()

# Interpret correlations
print("Trade-off Analysis:")
print(f"Strength vs. Ductility correlation: {correlation_matrix.loc['Strength', 'Ductility']:.3f}")
print(f"Strength vs. Cost correlation: {correlation_matrix.loc['Strength', 'Neg_Cost']:.3f}")
print(f"Ductility vs. Cost correlation: {correlation_matrix.loc['Ductility', 'Neg_Cost']:.3f}")
```

### Hypervolume Analysis

Track Pareto front quality using hypervolume:

```python
def calculate_hypervolume_2d(pareto_front, reference_point):
    """
    Calculate hypervolume for 2D Pareto front.
    
    Parameters:
    pareto_front: array-like, shape (n_points, 2)
    reference_point: array-like, shape (2,)
    
    Returns:
    hypervolume: float
    """
    # Sort points by first objective
    sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]
    
    hypervolume = 0
    prev_x = reference_point[0]
    
    for point in sorted_front:
        width = point[0] - prev_x
        height = point[1] - reference_point[1]
        hypervolume += width * height
        prev_x = point[0]
    
    return hypervolume

# Calculate hypervolume
reference_point = [150, 5]  # Slightly worse than worst known values
hv = calculate_hypervolume_2d(pareto_front[:, :2], reference_point)
print(f"Pareto front hypervolume: {hv:.2f}")
```


## Decision Making Strategies

### Weight-Based Selection

Select solutions based on preference weights:

```python
def weighted_selection(pareto_front, weights):
    """
    Select Pareto solution based on weighted sum.
    
    Parameters:
    pareto_front: array-like, shape (n_points, n_objectives)
    weights: array-like, shape (n_objectives,)
             Weights for each objective (should sum to 1)
    
    Returns:
    best_idx: int, index of best solution
    """
    # Normalize objectives to [0, 1] scale
    normalized_front = (pareto_front - pareto_front.min(axis=0)) / (pareto_front.max(axis=0) - pareto_front.min(axis=0))
    
    # Calculate weighted scores
    scores = np.dot(normalized_front, weights)
    
    return np.argmax(scores)

# Example: Prioritize strength (50%), ductility (30%), cost (20%)
weights = np.array([0.5, 0.3, 0.2])
best_idx = weighted_selection(pareto_front, weights)
selected_solution = pareto_front[best_idx]

print(f"Selected solution with weights {weights}:")
print(f"Strength: {selected_solution[0]:.1f} MPa")
print(f"Ductility: {selected_solution[1]:.1f} %")
print(f"Cost: {-selected_solution[2]:.1f} $/kg")  # Convert back from negative
```

### Interactive Selection

Allow users to interactively explore trade-offs:

```python
def interactive_selection(pareto_front, objective_names):
    """
    Interactive Pareto front exploration.
    """
    print("Pareto Front Solutions:")
    print("-" * 50)
    
    for i, solution in enumerate(pareto_front):
        print(f"Solution {i+1}:")
        for j, (obj_name, obj_value) in enumerate(zip(objective_names, solution)):
            print(f"  {obj_name}: {obj_value:.2f}")
        print()
    
    while True:
        try:
            choice = int(input(f"Select solution (1-{len(pareto_front)}) or 0 to exit: "))
            if choice == 0:
                break
            elif 1 <= choice <= len(pareto_front):
                selected = pareto_front[choice-1]
                print(f"\nSelected solution {choice}:")
                for obj_name, obj_value in zip(objective_names, selected):
                    print(f"  {obj_name}: {obj_value:.2f}")
                return choice-1
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    return None

# Example usage
objective_names = ['Strength (MPa)', 'Ductility (%)', 'Negative Cost ($/kg)']
# selected_idx = interactive_selection(pareto_front, objective_names)
```

### Multi-Criteria Decision Analysis (MCDA)

Use TOPSIS method for systematic decision making:

```python
def topsis_selection(pareto_front, weights, beneficial_criteria=None):
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    
    Parameters:
    pareto_front: array-like, shape (n_points, n_objectives)
    weights: array-like, shape (n_objectives,)
    beneficial_criteria: list of bool, True if objective should be maximized
    
    Returns:
    ranking: array of indices sorted by TOPSIS score (best first)
    """
    if beneficial_criteria is None:
        beneficial_criteria = [True] * pareto_front.shape[1]
    
    # Normalize decision matrix
    normalized = pareto_front / np.sqrt(np.sum(pareto_front**2, axis=0))
    
    # Apply weights
    weighted = normalized * weights
    
    # Determine ideal and negative-ideal solutions
    ideal_solution = np.zeros(weighted.shape[1])
    negative_ideal = np.zeros(weighted.shape[1])
    
    for j in range(weighted.shape[1]):
        if beneficial_criteria[j]:
            ideal_solution[j] = np.max(weighted[:, j])
            negative_ideal[j] = np.min(weighted[:, j])
        else:
            ideal_solution[j] = np.min(weighted[:, j])
            negative_ideal[j] = np.max(weighted[:, j])
    
    # Calculate distances
    dist_ideal = np.sqrt(np.sum((weighted - ideal_solution)**2, axis=1))
    dist_negative = np.sqrt(np.sum((weighted - negative_ideal)**2, axis=1))
    
    # Calculate TOPSIS scores
    scores = dist_negative / (dist_ideal + dist_negative)
    
    # Return ranking (best to worst)
    return np.argsort(scores)[::-1]

# Example: TOPSIS ranking
weights = np.array([0.4, 0.3, 0.3])
beneficial = [True, True, True]  # All objectives to be maximized
ranking = topsis_selection(pareto_front, weights, beneficial)

print("TOPSIS Ranking (best to worst):")
for i, idx in enumerate(ranking[:5]):  # Show top 5
    solution = pareto_front[idx]
    print(f"{i+1}. Strength: {solution[0]:.1f}, Ductility: {solution[1]:.1f}, Cost: {-solution[2]:.1f}")
```

## Advanced Pareto Analysis

### Pareto Front Approximation Quality

Assess how well your optimization approximates the true Pareto front:

```python
def pareto_front_metrics(pareto_front, reference_front=None):
    """
    Calculate Pareto front quality metrics.
    """
    metrics = {}
    
    # Number of solutions
    metrics['n_solutions'] = len(pareto_front)
    
    # Spread (diversity)
    if len(pareto_front) > 1:
        distances = []
        for i in range(len(pareto_front)):
            min_dist = float('inf')
            for j in range(len(pareto_front)):
                if i != j:
                    dist = np.linalg.norm(pareto_front[i] - pareto_front[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        metrics['min_distance'] = np.min(distances)
        metrics['avg_distance'] = np.mean(distances)
        metrics['spread'] = np.max(distances)
    
    # Hypervolume (if 2D or 3D)
    if pareto_front.shape[1] <= 3:
        ref_point = pareto_front.min(axis=0) - 0.1 * (pareto_front.max(axis=0) - pareto_front.min(axis=0))
        if pareto_front.shape[1] == 2:
            metrics['hypervolume'] = calculate_hypervolume_2d(pareto_front, ref_point)
    
    return metrics

# Calculate metrics
metrics = pareto_front_metrics(pareto_front)
print("Pareto Front Quality Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.3f}")
```

### Sensitivity Analysis

Analyze how sensitive the Pareto front is to changes in objectives:

```python
def sensitivity_analysis(pareto_front, perturbation=0.05):
    """
    Analyze sensitivity of Pareto front to objective perturbations.
    """
    original_hv = calculate_hypervolume_2d(pareto_front[:, :2], [150, 5])
    
    sensitivities = []
    for obj_idx in range(pareto_front.shape[1]):
        # Perturb objective
        perturbed_front = pareto_front.copy()
        perturbed_front[:, obj_idx] *= (1 + perturbation)
        
        # Recalculate Pareto front
        new_pareto_indices = find_pareto_front(perturbed_front)
        new_pareto_front = perturbed_front[new_pareto_indices]
        
        # Calculate new hypervolume
        if new_pareto_front.shape[1] >= 2:
            new_hv = calculate_hypervolume_2d(new_pareto_front[:, :2], [150, 5])
            sensitivity = (new_hv - original_hv) / original_hv
            sensitivities.append(sensitivity)
    
    return sensitivities

# Perform sensitivity analysis
sensitivities = sensitivity_analysis(pareto_front)
objective_names = ['Strength', 'Ductility', 'Cost']

print("Sensitivity Analysis (5% perturbation):")
for i, (obj_name, sensitivity) in enumerate(zip(objective_names, sensitivities)):
    print(f"{obj_name}: {sensitivity:.3f} (hypervolume change)")
```

## Practical Applications

### Alloy Design Case Study

Complete workflow for alloy optimization:

```python
# 1. Multi-objective optimization
VS_recommended, improvements, index = bgo.fit(
    'alloy_dataset.csv',
    'virtual_space.csv',
    object_num=3,
    method='EHVI',
    assign_model='GaussianProcess',
    bootstrap=10
)

# 2. Pareto front analysis
# Load training data and extract objectives
training_data = pd.read_csv('alloy_dataset.csv')
all_objectives = training_data[['Strength', 'Ductility', 'Neg_Cost']].values
pareto_indices = find_pareto_front(all_objectives)
pareto_front = all_objectives[pareto_indices]

# 3. Trade-off analysis
correlation_matrix = np.corrcoef(pareto_front.T)
print("Objective Correlations:")
print(correlation_matrix)

# 4. Decision making
weights = [0.4, 0.3, 0.3]  # Strength, ductility, cost
best_idx = weighted_selection(pareto_front, weights)
selected_alloy = pareto_front[best_idx]

print(f"\nRecommended Alloy Composition:")
print(f"Expected Strength: {selected_alloy[0]:.1f} MPa")
print(f"Expected Ductility: {selected_alloy[1]:.1f} %")
print(f"Expected Cost: {-selected_alloy[2]:.1f} $/kg")

# 5. Experimental validation
print(f"\nNext experiment: Virtual space index {index}")
print(f"Expected improvements: {improvements}")
```



## Next Steps

Now that you understand Pareto optimization:

1. **Practice with examples**: {doc}`examples/multi_objective`
2. **Try real materials cases**: {doc}`examples/materials_design`
3. **Learn advanced techniques**: {doc}`examples/advanced_usage`
4. **Explore visualization tools**: {doc}`visualization`

```{seealso}
For more on multi-objective optimization:
- Deb, K. "Multi-Objective Optimization using Evolutionary Algorithms"
- Miettinen, K. "Nonlinear Multiobjective Optimization"
- Zitzler, E. "Multiobjective Evolutionary Algorithms: A Comparative Case Study"
```
