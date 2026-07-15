# Batch Optimization Guide

## 🎯 Introduction

Batch optimization allows you to select multiple experiments to run in parallel, dramatically accelerating your optimization process. This is particularly valuable in materials science where you can synthesize multiple samples simultaneously or run parallel computational experiments.

## 🔄 Why Batch Optimization?

### Traditional Sequential Approach
```
Experiment 1 → Wait → Analyze → Experiment 2 → Wait → Analyze → ...
```

### Batch Parallel Approach
```
Experiments 1,2,3,4,5 → Wait → Analyze All → Next Batch → ...
```

**Benefits:**
- 🚀 **Faster convergence**: Reduce total optimization time
- 💰 **Cost efficiency**: Better resource utilization
- 🔬 **Practical**: Matches real experimental workflows
- 📊 **Better exploration**: Diverse parallel samples

## 📊 Available Batch Methods

### 1. Batch Expected Improvement (qEI)

Extends Expected Improvement to select multiple points that collectively maximize expected improvement.

```python
from Bgolearn import BGOsampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create comprehensive materials dataset
def create_materials_data():
    """Create realistic materials optimization dataset."""
    np.random.seed(42)
    
    # Define feature ranges (realistic for materials)
    n_samples = 40
    features = {
        'Temperature': np.random.uniform(300, 1200, n_samples),  # Kelvin
        'Pressure': np.random.uniform(0.1, 10, n_samples),      # GPa
        'Comp_A': np.random.uniform(0.1, 0.9, n_samples),      # Atomic fraction
        'Time': np.random.uniform(1, 100, n_samples)            # Hours
    }
    
    # Ensure composition constraint
    features['Comp_B'] = 1 - features['Comp_A']
    
    # Create complex target function (e.g., material strength)
    X = np.column_stack([features[k] for k in ['Temperature', 'Pressure', 'Comp_A', 'Comp_B', 'Time']])
    
    # Complex non-linear relationship
    temp_effect = (X[:, 0] - 600) / 600  # Optimal around 600K
    pressure_effect = np.log(X[:, 1] + 0.1)
    comp_effect = 4 * X[:, 2] * X[:, 3]  # Interaction term
    time_effect = np.sqrt(X[:, 4])
    
    y = (10 - temp_effect**2 + 2*pressure_effect + 
         comp_effect + 0.1*time_effect + 
         0.2*np.random.randn(n_samples))
    
    df = pd.DataFrame(X, columns=['Temperature', 'Pressure', 'Comp_A', 'Comp_B', 'Time'])
    df['Strength'] = y
    
    return df

# Create candidate space
def create_candidate_space(n_candidates=500):
    """Create diverse candidate materials."""
    np.random.seed(123)
    
    candidates = []
    for _ in range(n_candidates):
        temp = np.random.uniform(300, 1200)
        pressure = np.random.uniform(0.1, 10)
        comp_a = np.random.uniform(0.1, 0.9)
        comp_b = 1 - comp_a
        time = np.random.uniform(1, 100)
        
        candidates.append([temp, pressure, comp_a, comp_b, time])
    
    return pd.DataFrame(candidates, 
                       columns=['Temperature', 'Pressure', 'Comp_A', 'Comp_B', 'Time'])

# Prepare data
train_data = create_materials_data()
X_train = train_data.drop('Strength', axis=1)
y_train = train_data['Strength']
X_candidates = create_candidate_space()

print(f"Training data: {X_train.shape}")
print(f"Candidates: {X_candidates.shape}")
print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")

# Fit model
optimizer = BGOsampling.Bgolearn()
model = optimizer.fit(
    data_matrix=X_train,
    Measured_response=y_train,
    virtual_samples=X_candidates,
    min_search=False,  # Maximize strength
    CV_test=5,
    Normalize=True
)

print("Model fitted successfully!")
```

### Basic Batch EI Usage

```python
# Batch Expected Improvement - select 5 experiments
print("\n=== Batch Expected Improvement ===")

# Small batch for quick testing
batch_small = model.batch_EI(q=3, mc_samples=500)
batch_indices_small, batch_points_small = batch_small

print(f"Small batch selected {len(batch_points_small)} experiments:")
for i, point in enumerate(batch_points_small):
    print(f"  Experiment {i+1}: T={point[0]:.1f}K, P={point[1]:.2f}GPa, "
          f"Comp_A={point[2]:.3f}, Time={point[4]:.1f}h")

# Larger batch for parallel processing
batch_large = model.batch_EI(q=8, mc_samples=1000)
batch_indices_large, batch_points_large = batch_large

print(f"\nLarge batch selected {len(batch_points_large)} experiments:")
for i, point in enumerate(batch_points_large):
    print(f"  Experiment {i+1}: T={point[0]:.1f}K, P={point[1]:.2f}GPa, "
          f"Comp_A={point[2]:.3f}, Time={point[4]:.1f}h")
```

### Advanced Batch EI Parameters

```python
# High-precision batch selection
print("\n=== High-Precision Batch EI ===")
batch_precise = model.batch_EI(q=5, mc_samples=2000)
batch_indices_precise, batch_points_precise = batch_precise

# Compare with different MC sample sizes
mc_samples_list = [100, 500, 1000, 2000]
batch_comparison = {}

for mc_samples in mc_samples_list:
    batch_result = model.batch_EI(q=3, mc_samples=mc_samples)
    batch_comparison[mc_samples] = batch_result[1]  # Store points
    print(f"MC samples {mc_samples}: First point T={batch_result[1][0][0]:.1f}K")
```

### 2. Batch Upper Confidence Bound (qUCB)

Selects multiple points with high upper confidence bounds.

```python
# Batch Upper Confidence Bound
print("\n=== Batch Upper Confidence Bound ===")

# Conservative exploration
batch_ucb_conservative = model.batch_UCB(q=5, beta=1.5)
indices_conservative, points_conservative = batch_ucb_conservative

print("Conservative UCB (β=1.5):")
for i, point in enumerate(points_conservative):
    print(f"  Experiment {i+1}: T={point[0]:.1f}K, P={point[1]:.2f}GPa")

# Balanced exploration
batch_ucb_balanced = model.batch_UCB(q=5, beta=2.0)
indices_balanced, points_balanced = batch_ucb_balanced

print("\nBalanced UCB (β=2.0):")
for i, point in enumerate(points_balanced):
    print(f"  Experiment {i+1}: T={point[0]:.1f}K, P={point[1]:.2f}GPa")

# Aggressive exploration
batch_ucb_aggressive = model.batch_UCB(q=5, beta=3.0)
indices_aggressive, points_aggressive = batch_ucb_aggressive

print("\nAggressive UCB (β=3.0):")
for i, point in enumerate(points_aggressive):
    print(f"  Experiment {i+1}: T={point[0]:.1f}K, P={point[1]:.2f}GPa")
```

### 3. Advanced Batch Functions

Using the standalone batch acquisition module:

```python
from bgolearn.batch_acquisition import (
    BatchAcquisitionFunctions,
    parallel_expected_improvement,
    parallel_upper_confidence_bound
)

# Create batch acquisition object
batch_acq = BatchAcquisitionFunctions(
    model=model.Kriging_model(),
    X_train=X_train.values,
    y_train=y_train.values,
    minimize_objective=False  # We want to maximize strength
)

# Direct batch selection
print("\n=== Advanced Batch Methods ===")

# Method 1: Direct qEI
qei_values = batch_acq.q_expected_improvement(X_candidates.values, q=4, mc_samples=1000)
print(f"qEI values range: [{qei_values.min():.4f}, {qei_values.max():.4f}]")

# Method 2: Direct qUCB
qucb_values = batch_acq.q_upper_confidence_bound(X_candidates.values, q=4, beta=2.0)
print(f"qUCB values range: [{qucb_values.min():.4f}, {qucb_values.max():.4f}]")

# Method 3: Greedy batch selection
greedy_indices, greedy_points = batch_acq.select_batch_greedy(
    X_candidates.values, q=6, acquisition_func='qei', mc_samples=1000
)

print(f"Greedy selection chose {len(greedy_points)} points:")
for i, (idx, point) in enumerate(zip(greedy_indices, greedy_points)):
    print(f"  Point {i+1} (index {idx}): T={point[0]:.1f}K, P={point[1]:.2f}GPa")
```

### 4. Convenience Functions

```python
# Convenience functions for quick batch optimization
print("\n=== Convenience Functions ===")

# Parallel Expected Improvement
pei_indices, pei_points = parallel_expected_improvement(
    X_candidates.values,
    model.Kriging_model(),
    X_train.values,
    y_train.values,
    q=4,
    minimize_objective=False
)

print(f"Parallel EI selected {len(pei_points)} points")

# Parallel Upper Confidence Bound
pucb_indices, pucb_points = parallel_upper_confidence_bound(
    X_candidates.values,
    model.Kriging_model(),
    X_train.values,
    y_train.values,
    q=4,
    beta=2.5,
    minimize_objective=False
)

print(f"Parallel UCB selected {len(pucb_points)} points")
```

## 📊 Batch Size Selection

### Optimal Batch Size Analysis

```python
def analyze_batch_sizes():
    """Analyze the effect of different batch sizes."""
    batch_sizes = [1, 2, 3, 5, 8, 10, 15, 20]
    results = {}
    
    print("Batch Size Analysis:")
    print("-" * 50)
    
    for q in batch_sizes:
        if q > len(X_candidates):
            continue
            
        try:
            # Time the batch selection
            import time
            start_time = time.time()
            
            batch_result = model.batch_EI(q=q, mc_samples=500)
            
            end_time = time.time()
            computation_time = end_time - start_time
            
            results[q] = {
                'points': batch_result[1],
                'time': computation_time,
                'diversity': calculate_diversity(batch_result[1])
            }
            
            print(f"Batch size {q:2d}: {computation_time:.2f}s, "
                  f"diversity: {results[q]['diversity']:.3f}")
            
        except Exception as e:
            print(f"Batch size {q:2d}: Failed ({e})")
    
    return results

def calculate_diversity(points):
    """Calculate diversity of selected points."""
    if len(points) < 2:
        return 0.0
    
    points_array = np.array(points)
    distances = []
    
    for i in range(len(points_array)):
        for j in range(i+1, len(points_array)):
            dist = np.linalg.norm(points_array[i] - points_array[j])
            distances.append(dist)
    
    return np.mean(distances)

# Run analysis
batch_analysis = analyze_batch_sizes()
```

### Batch Size Recommendations

```python
def recommend_batch_size(n_candidates, n_features, available_resources):
    """Recommend optimal batch size based on problem characteristics."""
    
    # Base recommendations
    if n_candidates < 50:
        base_size = min(3, n_candidates // 10)
    elif n_candidates < 200:
        base_size = min(5, n_candidates // 20)
    elif n_candidates < 1000:
        base_size = min(10, n_candidates // 50)
    else:
        base_size = min(20, n_candidates // 100)
    
    # Adjust for dimensionality
    if n_features > 10:
        base_size = min(base_size * 2, 15)
    
    # Adjust for resources
    resource_factor = min(available_resources / 5, 2.0)
    recommended_size = int(base_size * resource_factor)
    
    return max(1, min(recommended_size, available_resources))

# Get recommendation
n_features = X_train.shape[1]
n_candidates = len(X_candidates)
available_resources = 10  # Number of parallel experiments you can run

recommended_q = recommend_batch_size(n_candidates, n_features, available_resources)
print(f"\nRecommended batch size: {recommended_q}")

# Use recommended size
optimal_batch = model.batch_EI(q=recommended_q, mc_samples=1000)
print(f"Selected {len(optimal_batch[1])} experiments with recommended batch size")
```

## 🎨 Visualizing Batch Selection

### 2D Visualization

```python
def visualize_batch_selection_2d():
    """Visualize batch selection in 2D feature space."""
    
    # Create 2D problem for visualization
    np.random.seed(42)
    
    # Generate 2D training data
    X_2d = np.random.uniform(-3, 3, (15, 2))
    y_2d = -(X_2d[:, 0]**2 + X_2d[:, 1]**2) + 0.2 * np.random.randn(15)
    
    # Create 2D candidate grid
    x1 = np.linspace(-3, 3, 30)
    x2 = np.linspace(-3, 3, 30)
    X1, X2 = np.meshgrid(x1, x2)
    X_candidates_2d = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Fit 2D model
    optimizer_2d = BGOsampling.Bgolearn()
    model_2d = optimizer_2d.fit(
        pd.DataFrame(X_2d, columns=['x1', 'x2']),
        pd.Series(y_2d),
        pd.DataFrame(X_candidates_2d, columns=['x1', 'x2']),
        min_search=False
    )
    
    # Get batch selection
    batch_2d = model_2d.batch_EI(q=5, mc_samples=500)
    batch_points_2d = batch_2d[1]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training data and batch selection
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap='viridis', 
               s=100, edgecolors='black', label='Training data')
    ax1.scatter([p[0] for p in batch_points_2d], 
               [p[1] for p in batch_points_2d],
               c='red', s=150, marker='*', edgecolors='white', linewidth=2,
               label='Batch selection')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('Batch Selection in Feature Space')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Acquisition function landscape
    ei_values_2d, _ = model_2d.EI()
    ei_grid = ei_values_2d.reshape(30, 30)
    
    im = ax2.contourf(X1, X2, ei_grid, levels=20, cmap='plasma', alpha=0.8)
    ax2.contour(X1, X2, ei_grid, levels=10, colors='white', alpha=0.5, linewidths=0.5)
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c='white', s=80, edgecolors='black')
    ax2.scatter([p[0] for p in batch_points_2d], 
               [p[1] for p in batch_points_2d],
               c='red', s=150, marker='*', edgecolors='white', linewidth=2)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('EI Landscape with Batch Selection')
    plt.colorbar(im, ax=ax2, label='Expected Improvement')
    
    plt.tight_layout()
    plt.show()

# Run 2D visualization
visualize_batch_selection_2d()
```

### Batch Diversity Analysis

```python
def analyze_batch_diversity():
    """Analyze diversity of batch selections."""
    
    # Compare different batch methods
    methods = {
        'Batch EI': lambda: model.batch_EI(q=6, mc_samples=500),
        'Batch UCB (β=1.5)': lambda: model.batch_UCB(q=6, beta=1.5),
        'Batch UCB (β=2.5)': lambda: model.batch_UCB(q=6, beta=2.5),
        'Batch UCB (β=3.5)': lambda: model.batch_UCB(q=6, beta=3.5)
    }
    
    diversity_results = {}
    
    for method_name, method_func in methods.items():
        try:
            _, points = method_func()
            diversity = calculate_diversity(points)
            diversity_results[method_name] = {
                'points': points,
                'diversity': diversity
            }
        except Exception as e:
            print(f"Error with {method_name}: {e}")
    
    # Plot diversity comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Diversity bar plot
    methods_list = list(diversity_results.keys())
    diversities = [diversity_results[m]['diversity'] for m in methods_list]
    
    bars = ax1.bar(range(len(methods_list)), diversities, 
                   color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Average Pairwise Distance')
    ax1.set_title('Batch Selection Diversity')
    ax1.set_xticks(range(len(methods_list)))
    ax1.set_xticklabels(methods_list, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, diversity in zip(bars, diversities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{diversity:.3f}', ha='center', va='bottom')
    
    # Feature space coverage (Temperature vs Pressure)
    colors = ['blue', 'green', 'orange', 'red']
    for i, (method_name, result) in enumerate(diversity_results.items()):
        points = result['points']
        temps = [p[0] for p in points]
        pressures = [p[1] for p in points]
        ax2.scatter(temps, pressures, c=colors[i], label=method_name, 
                   s=100, alpha=0.7, edgecolors='black')
    
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Pressure (GPa)')
    ax2.set_title('Batch Selection in Temperature-Pressure Space')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return diversity_results

# Run diversity analysis
diversity_analysis = analyze_batch_diversity()
```

## 🔧 Practical Implementation

### Iterative Batch Optimization

```python
def iterative_batch_optimization(initial_model, n_iterations=3, batch_size=4):
    """Demonstrate iterative batch optimization workflow."""
    
    current_model = initial_model
    all_experiments = []
    convergence_history = []
    
    print("Iterative Batch Optimization")
    print("=" * 50)
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}:")
        
        # Select batch
        batch_result = current_model.batch_EI(q=batch_size, mc_samples=1000)
        batch_indices, batch_points = batch_result
        
        print(f"Selected {len(batch_points)} experiments:")
        for i, point in enumerate(batch_points):
            print(f"  Exp {i+1}: T={point[0]:.1f}K, P={point[1]:.2f}GPa, "
                  f"Comp_A={point[2]:.3f}")
        
        # Simulate experimental results (in practice, you'd run real experiments)
        simulated_results = simulate_experiments(batch_points)
        
        # Update training data
        new_X = pd.DataFrame(batch_points, columns=X_train.columns)
        new_y = pd.Series(simulated_results)
        
        updated_X = pd.concat([X_train, new_X], ignore_index=True)
        updated_y = pd.concat([y_train, new_y], ignore_index=True)
        
        # Refit model with new data
        optimizer_updated = BGOsampling.Bgolearn()
        current_model = optimizer_updated.fit(
            data_matrix=updated_X,
            Measured_response=updated_y,
            virtual_samples=X_candidates,
            min_search=False,
            Normalize=True
        )
        
        # Track progress
        current_best = updated_y.max()
        convergence_history.append(current_best)
        all_experiments.extend(batch_points)
        
        print(f"Current best: {current_best:.3f}")
        print(f"Total experiments: {len(all_experiments)}")
    
    return current_model, all_experiments, convergence_history

def simulate_experiments(batch_points):
    """Simulate experimental results for demonstration."""
    results = []
    for point in batch_points:
        # Use the same function as data generation with some noise
        temp_effect = (point[0] - 600) / 600
        pressure_effect = np.log(point[1] + 0.1)
        comp_effect = 4 * point[2] * point[3]
        time_effect = np.sqrt(point[4])
        
        result = (10 - temp_effect**2 + 2*pressure_effect + 
                 comp_effect + 0.1*time_effect + 
                 0.3*np.random.randn())  # More noise for realism
        results.append(result)
    
    return results

# Run iterative optimization
final_model, all_experiments, history = iterative_batch_optimization(
    model, n_iterations=3, batch_size=4
)

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(history) + 1), history, 'o-', linewidth=2, markersize=8)
plt.xlabel('Iteration')
plt.ylabel('Best Observed Value')
plt.title('Batch Optimization Convergence')
plt.grid(True, alpha=0.3)
plt.show()
```

## 💡 Best Practices

### 1. Batch Size Selection

```python
# Guidelines for batch size selection
def batch_size_guidelines():
    """Provide guidelines for batch size selection."""
    
    guidelines = {
        "Small problems (< 100 candidates)": "Use q = 2-3",
        "Medium problems (100-1000 candidates)": "Use q = 3-8", 
        "Large problems (> 1000 candidates)": "Use q = 5-20",
        "High-dimensional (> 10 features)": "Increase batch size by 50%",
        "Noisy functions": "Use smaller batches (q = 2-5)",
        "Expensive experiments": "Use larger batches to amortize cost",
        "Limited resources": "Match batch size to available resources"
    }
    
    print("Batch Size Selection Guidelines:")
    print("-" * 40)
    for condition, recommendation in guidelines.items():
        print(f"{condition}: {recommendation}")

batch_size_guidelines()
```

### 2. Quality Control

```python
def batch_quality_control(batch_points, X_train):
    """Perform quality control on batch selection."""
    
    issues = []
    
    # Check for duplicates
    batch_array = np.array(batch_points)
    for i in range(len(batch_array)):
        for j in range(i+1, len(batch_array)):
            if np.allclose(batch_array[i], batch_array[j], rtol=1e-3):
                issues.append(f"Near-duplicate points: {i} and {j}")
    
    # Check distance to training data
    min_distances = []
    for point in batch_points:
        distances = [np.linalg.norm(np.array(point) - np.array(train_point)) 
                    for train_point in X_train.values]
        min_distances.append(min(distances))
    
    if any(d < 0.1 for d in min_distances):
        issues.append("Some points very close to training data")
    
    # Check feature ranges
    for i, col in enumerate(X_train.columns):
        batch_values = [point[i] for point in batch_points]
        train_min, train_max = X_train[col].min(), X_train[col].max()
        
        if any(v < train_min or v > train_max for v in batch_values):
            issues.append(f"Points outside training range for {col}")
    
    if issues:
        print("Quality Control Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Batch selection passed quality control")
    
    return len(issues) == 0

# Test quality control
batch_test = model.batch_EI(q=5, mc_samples=500)
is_good_batch = batch_quality_control(batch_test[1], X_train)
```

### 3. Resource Planning

```python
def plan_experimental_resources(batch_points, cost_per_experiment=100, 
                              time_per_experiment=2):
    """Plan resources for batch experiments."""
    
    n_experiments = len(batch_points)
    total_cost = n_experiments * cost_per_experiment
    total_time = time_per_experiment  # Parallel execution
    
    print("Experimental Resource Planning:")
    print("-" * 35)
    print(f"Number of experiments: {n_experiments}")
    print(f"Cost per experiment: ${cost_per_experiment}")
    print(f"Total cost: ${total_cost}")
    print(f"Time per experiment: {time_per_experiment} hours")
    print(f"Total time (parallel): {total_time} hours")
    print(f"Time savings vs sequential: {n_experiments * time_per_experiment - total_time} hours")
    
    return {
        'n_experiments': n_experiments,
        'total_cost': total_cost,
        'total_time': total_time,
        'time_savings': n_experiments * time_per_experiment - total_time
    }

# Plan resources for our batch
resource_plan = plan_experimental_resources(batch_test[1])
```
