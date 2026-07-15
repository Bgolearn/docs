# Optimization Strategies

```{note}
This page covers advanced optimization strategies and best practices for getting the most out of Bgolearn.
```

## Overview

Effective optimization with Bgolearn requires understanding not just the algorithms, but also the strategies for applying them to real-world problems. This page covers advanced techniques, best practices, and common pitfalls to avoid.

## Sequential vs. Batch Optimization

### Sequential Optimization

The traditional approach where experiments are conducted one at a time:

```python
from Bgolearn import BGOsampling

# Sequential optimization loop
opt = BGOsampling.Bgolearn()

for iteration in range(10):
    # Fit model and get recommendation
    model = opt.fit(
        data_matrix=current_data,
        Measured_response=current_response,
        virtual_samples=virtual_samples,
        opt_num=1  # Single recommendation
    )
    
    # Conduct experiment
    ei_values, recommended_points = model.EI()
    new_x = recommended_points[0]
    new_y = conduct_experiment(new_x)
    
    # Update dataset
    current_data = np.vstack([current_data, new_x])
    current_response = np.append(current_response, new_y)
```

**Advantages:**
- Maximum information gain per experiment
- Adaptive to new information
- Lower computational cost per iteration

**Disadvantages:**
- Slower overall progress
- Cannot utilize parallel experimental capabilities
- Higher overhead per experiment

### Batch Optimization

Selecting multiple experiments simultaneously:

```python
# Batch optimization
model = opt.fit(
    data_matrix=current_data,
    Measured_response=current_response,
    virtual_samples=virtual_samples,
    opt_num=5  # Multiple recommendations
)

# Get batch of experiments
batch_indices = model['recommended_indices']
batch_experiments = virtual_samples[batch_indices]

# Conduct experiments in parallel
batch_results = parallel_experiments(batch_experiments)
```

**Advantages:**
- Faster overall progress
- Utilizes parallel experimental capabilities
- Better resource utilization

**Disadvantages:**
- Less adaptive within batch
- Higher computational cost
- Potential redundancy in batch

## Exploration vs. Exploitation Strategies

### Exploration-Focused Strategies

When you need to explore the design space thoroughly:

```python
# High exploration settings
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Classifier='GaussianProcess',  # Good uncertainty estimates
    # Use acquisition functions that favor exploration
    Dynamic_W=True,  # Dynamic weighting
    seed=42
)
```

**Use when:**
- Early stages of optimization
- Large, unexplored design spaces
- High uncertainty in measurements
- Discovery-focused research

### Exploitation-Focused Strategies

When you want to refine known good regions:

```python
# High exploitation settings
# Focus virtual space around known good regions
good_region_mask = (virtual_samples[:, 0] > best_composition[0] - 0.5) & \
                   (virtual_samples[:, 0] < best_composition[0] + 0.5)
focused_virtual = virtual_samples[good_region_mask]

model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=focused_virtual,
    opt_num=3  # Multiple points in good region
)
```

**Use when:**
- Late stages of optimization
- Well-understood systems
- Limited experimental budget
- Optimization-focused research

## Adaptive Strategies

### Dynamic Virtual Space

Adapt the virtual space based on optimization progress:

```python
def adaptive_virtual_space(iteration, best_point, initial_space):
    """Adapt virtual space based on optimization progress."""
    if iteration < 5:
        # Early: broad exploration
        return initial_space
    elif iteration < 15:
        # Middle: focus around promising regions
        distances = np.linalg.norm(initial_space - best_point, axis=1)
        close_mask = distances < np.percentile(distances, 50)
        return initial_space[close_mask]
    else:
        # Late: local refinement
        distances = np.linalg.norm(initial_space - best_point, axis=1)
        close_mask = distances < np.percentile(distances, 25)
        return initial_space[close_mask]

# Use in optimization loop
for iteration in range(20):
    current_virtual = adaptive_virtual_space(iteration, best_so_far, full_virtual_space)
    model = opt.fit(virtual_samples=current_virtual, ...)
```

### Multi-Stage Optimization

Different strategies for different phases:

```python
def multi_stage_optimization(data_matrix, measured_response, virtual_samples):
    """Multi-stage optimization strategy."""
    
    # Stage 1: Broad exploration (Random Forest for robustness)
    print("Stage 1: Exploration")
    for i in range(5):
        model = opt.fit(
            data_matrix=data_matrix,
            Measured_response=measured_response,
            virtual_samples=virtual_samples,
            Classifier='RandomForest',
            opt_num=2
        )
        # Add new points...
    
    # Stage 2: Focused search (Gaussian Process for uncertainty)
    print("Stage 2: Focused Search")
    for i in range(10):
        model = opt.fit(
            data_matrix=data_matrix,
            Measured_response=measured_response,
            virtual_samples=virtual_samples,
            Classifier='GaussianProcess',
            opt_num=1
        )
        # Add new points...
    
    # Stage 3: Local refinement
    print("Stage 3: Local Refinement")
    # Focus on best region...
```

## Model Selection Strategies

### Ensemble Approaches

Combine multiple models for robust predictions:

```python
def ensemble_optimization(data_matrix, measured_response, virtual_samples):
    """Use ensemble of models for robust optimization."""
    
    models = ['GaussianProcess', 'RandomForest', 'SVR']
    recommendations = []
    
    for model_name in models:
        model = opt.fit(
            data_matrix=data_matrix,
            Measured_response=measured_response,
            virtual_samples=virtual_samples,
            Classifier=model_name,
            opt_num=3
        )
        recommendations.extend(model['recommended_indices'])
    
    # Remove duplicates and select diverse set
    unique_recommendations = list(set(recommendations))
    return unique_recommendations[:5]  # Top 5 diverse recommendations
```

### Adaptive Model Selection

Choose models based on problem characteristics:

```python
def adaptive_model_selection(data_matrix, measured_response):
    """Select model based on data characteristics."""
    
    n_samples, n_features = data_matrix.shape
    noise_level = np.std(measured_response) / np.mean(measured_response)
    
    if n_samples < 20:
        return 'GaussianProcess'  # Good for small data
    elif n_features > 10:
        return 'RandomForest'    # Good for high dimensions
    elif noise_level > 0.2:
        return 'SVR'            # Good for noisy data
    else:
        return 'GaussianProcess' # Default choice
```

## Constraint Handling Strategies

### Soft Constraints

Handle constraints through penalty methods:

```python
def apply_soft_constraints(virtual_samples, constraints):
    """Apply soft constraints via penalty method."""
    
    penalties = np.zeros(len(virtual_samples))
    
    for i, sample in enumerate(virtual_samples):
        penalty = 0
        
        # Composition constraint
        if np.sum(sample[:3]) > 7.0:
            penalty += 1000 * (np.sum(sample[:3]) - 7.0)
        
        # Ratio constraint
        if sample[1] > 0:  # Avoid division by zero
            ratio = sample[0] / sample[1]
            if ratio < 1.5 or ratio > 4.0:
                penalty += 1000 * abs(ratio - np.clip(ratio, 1.5, 4.0))
        
        penalties[i] = penalty
    
    return penalties

# Use in optimization
penalties = apply_soft_constraints(virtual_samples, constraints)
# Modify acquisition function to include penalties
```

### Hard Constraints

Filter virtual space to satisfy constraints:

```python
def apply_hard_constraints(virtual_samples):
    """Filter virtual samples to satisfy hard constraints."""
    
    valid_mask = np.ones(len(virtual_samples), dtype=bool)
    
    for i, sample in enumerate(virtual_samples):
        # Check all constraints
        if np.sum(sample[:3]) > 7.0:
            valid_mask[i] = False
        if sample[1] > 0 and not (1.5 <= sample[0]/sample[1] <= 4.0):
            valid_mask[i] = False
        if sample[2] < 0.2:
            valid_mask[i] = False
    
    return virtual_samples[valid_mask]

# Use in optimization
feasible_virtual = apply_hard_constraints(virtual_samples)
model = opt.fit(virtual_samples=feasible_virtual, ...)
```

## Uncertainty Quantification Strategies

### Bootstrap Ensemble

Use multiple model fits for uncertainty estimation:

```python
def bootstrap_uncertainty(data_matrix, measured_response, virtual_samples, n_bootstrap=10):
    """Estimate uncertainty using bootstrap ensemble."""
    
    predictions = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        n_samples = len(data_matrix)
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        
        bootstrap_data = data_matrix[bootstrap_indices]
        bootstrap_response = measured_response[bootstrap_indices]
        
        # Fit model
        model = opt.fit(
            data_matrix=bootstrap_data,
            Measured_response=bootstrap_response,
            virtual_samples=virtual_samples,
            Classifier='GaussianProcess'
        )
        
        predictions.append(model.virtual_samples_mean)
    
    # Calculate statistics
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred
```

## Performance Optimization

### Computational Efficiency

Strategies for faster optimization:

```python
# 1. Reduce virtual space size
def smart_virtual_space_reduction(virtual_samples, current_best, reduction_factor=0.5):
    """Intelligently reduce virtual space size."""
    
    # Calculate distances from current best
    distances = np.linalg.norm(virtual_samples - current_best, axis=1)
    
    # Keep closest points and some random distant points
    n_keep = int(len(virtual_samples) * reduction_factor)
    n_close = int(n_keep * 0.7)
    n_random = n_keep - n_close
    
    # Closest points
    close_indices = np.argsort(distances)[:n_close]
    
    # Random distant points
    distant_indices = np.argsort(distances)[n_close:]
    random_distant = np.random.choice(distant_indices, n_random, replace=False)
    
    selected_indices = np.concatenate([close_indices, random_distant])
    return virtual_samples[selected_indices]

# 2. Use faster models for initial screening
def hierarchical_optimization(data_matrix, measured_response, virtual_samples):
    """Use fast models for screening, then GP for refinement."""
    
    # Stage 1: Fast screening with Random Forest
    model_rf = opt.fit(
        data_matrix=data_matrix,
        Measured_response=measured_response,
        virtual_samples=virtual_samples,
        Classifier='RandomForest',
        opt_num=50  # Get many candidates
    )
    
    # Stage 2: Refine with Gaussian Process
    top_candidates = virtual_samples[model_rf['recommended_indices']]
    
    model_gp = opt.fit(
        data_matrix=data_matrix,
        Measured_response=measured_response,
        virtual_samples=top_candidates,
        Classifier='GaussianProcess',
        opt_num=5  # Final selection
    )
    
    return model_gp
```

## Convergence Monitoring

### Tracking Optimization Progress

```python
def monitor_convergence(optimization_history):
    """Monitor and analyze optimization convergence."""
    
    best_values = []
    improvements = []
    
    for i, result in enumerate(optimization_history):
        if i == 0:
            best_values.append(result)
            improvements.append(0)
        else:
            current_best = max(best_values[-1], result)
            best_values.append(current_best)
            improvements.append(current_best - best_values[-2])
    
    # Check convergence criteria
    recent_improvements = improvements[-5:]  # Last 5 iterations
    avg_improvement = np.mean(recent_improvements)
    
    converged = avg_improvement < 0.01  # Threshold
    
    return {
        'best_values': best_values,
        'improvements': improvements,
        'converged': converged,
        'avg_recent_improvement': avg_improvement
    }
```

## Best Practices Summary

### Data Quality
1. **Sufficient Data**: >10 samples per feature
2. **Representative Sampling**: Cover design space well
3. **Quality Control**: Remove outliers and errors
4. **Validation**: Use cross-validation

### Model Selection
1. **Start Simple**: Begin with Gaussian Process
2. **Consider Data Size**: RF for large data, GP for small
3. **Handle Noise**: SVR for noisy data
4. **Validate Performance**: Compare multiple models

### Optimization Strategy
1. **Multi-Stage Approach**: Exploration → Focused → Refinement
2. **Adaptive Virtual Space**: Adjust based on progress
3. **Constraint Handling**: Choose hard vs. soft constraints
4. **Uncertainty Quantification**: Use bootstrap or GP uncertainty

### Computational Efficiency
1. **Smart Virtual Space**: Reduce size intelligently
2. **Hierarchical Models**: Fast screening + accurate refinement
3. **Parallel Experiments**: Use batch optimization
4. **Monitor Convergence**: Stop when converged

## Troubleshooting Common Issues

### Poor Convergence
- Increase exploration in early stages
- Check data quality and preprocessing
- Try different surrogate models
- Expand virtual space

### Slow Performance
- Reduce virtual space size
- Use simpler models (RandomForest)
- Implement hierarchical optimization
- Parallelize computations

### Constraint Violations
- Use hard constraints for critical limits
- Implement soft constraints for preferences
- Validate constraint definitions
- Check feasible region size

## Next Steps

- **Practice with examples**: {doc}`examples/single_objective`
- **Learn multi-objective strategies**: {doc}`multibgolearn`
- **Explore advanced applications**: {doc}`materials_discovery`
- **Try batch optimization**: {doc}`batch_optimization`
