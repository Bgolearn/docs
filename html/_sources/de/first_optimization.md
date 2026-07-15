# Your First Optimization

```{note}
Diese Seite ist die deutsche Version des Bgolearn-Handbuchs.
```

```{note}
This tutorial walks you through your first Bayesian optimization with Bgolearn step by step.
```

## Problem Setup

Let's optimize a simple 1D function to understand the workflow. We'll find the minimum of:

$$f(x) = (x-2)^2 + 0.1 \sin(10x)$$

This function has a global minimum near $x ≈ 1.75$ (the sine term shifts it slightly from $x = 2$) and includes oscillations to make it interesting.

## Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bgolearn import BGOsampling

# Set random seed for reproducibility
np.random.seed(42)
```

## Step 2: Define the Objective Function

```python
def objective_function(x):
    """
    Our test function to minimize: f(x) = (x-2)^2 + 0.1*sin(10x)
    Global minimum at x ≈ 1.77 (sine term shifts it from x=2).
    """
    return (x - 2)**2 + 0.1 * np.sin(10 * x)

# Visualize the function
x_plot = np.linspace(0, 4, 200)
y_plot = [objective_function(x) for x in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Objective function')
plt.axvline(x=1.75, color='red', linestyle='--', alpha=0.7, label='True minimum')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Objective Function to Minimize')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"True minimum: f(1.75) = {objective_function(1.75):.4f}")
```

## Step 3: Generate Initial Data

In real optimization, we start with a few initial experiments:

```python
# Generate initial training data
n_initial = 5
X_train = np.random.uniform(0, 4, n_initial).reshape(-1, 1)
y_train = np.array([objective_function(x[0]) for x in X_train])

# Add some experimental noise
noise_level = 0.02
y_train += noise_level * np.random.randn(len(y_train))

# Convert to pandas (Bgolearn's preferred format)
X_train_df = pd.DataFrame(X_train, columns=['x'])
y_train_series = pd.Series(y_train)

print(f"Initial data: {len(X_train_df)} points")
print(f"Current best: f({X_train_df.iloc[y_train_series.argmin()]['x']:.3f}) = {y_train_series.min():.4f}")
```

## Step 4: Define Candidate Points

Create the search space - where we might sample next:

```python
# Generate candidate points
X_candidates = np.linspace(0, 4, 100).reshape(-1, 1)
X_candidates_df = pd.DataFrame(X_candidates, columns=['x'])

print(f"Search space: {len(X_candidates_df)} candidate points")
```

## Step 5: Fit the Bgolearn Model

```python
# Initialize and fit Bgolearn
optimizer = BGOsampling.Bgolearn()

model = optimizer.fit(
    data_matrix=X_train_df,
    Measured_response=y_train_series,
    virtual_samples=X_candidates_df,
    min_search=True,  # We want to minimize
    CV_test=3,        # 3-fold cross-validation
    Normalize=True    # Normalize features
)

print("Bgolearn model fitted successfully!")
```

## Step 6: Visualize the Model

```python
# Get model predictions
mean_pred = model.virtual_samples_mean
std_pred = model.virtual_samples_std

# Create visualization
plt.figure(figsize=(12, 8))

# Plot true function
plt.plot(x_plot, y_plot, 'b-', linewidth=2, alpha=0.7, label='True function')

# Plot GP mean prediction
plt.plot(X_candidates, mean_pred, 'g-', linewidth=2, label='GP mean')

# Plot uncertainty (±2σ)
plt.fill_between(X_candidates.flatten(), 
                mean_pred - 2*std_pred,
                mean_pred + 2*std_pred,
                alpha=0.3, color='green', label='GP uncertainty (±2σ)')

# Plot training data
plt.scatter(X_train, y_train, c='red', s=100, zorder=5, 
           edgecolors='black', linewidth=1, label='Training data')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gaussian Process Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Step 7: Use Acquisition Functions

Find the next best point to sample:

```python
# Expected Improvement
ei_values, ei_point = model.EI()
print(f"Expected Improvement recommends: x = {ei_point[0][0]:.3f}")

# Upper Confidence Bound
ucb_values, ucb_point = model.UCB(alpha=2.0)
print(f"Upper Confidence Bound recommends: x = {ucb_point[0][0]:.3f}")

# Probability of Improvement
poi_values, poi_point = model.PoI(tao=0.01)
print(f"Probability of Improvement recommends: x = {poi_point[0][0]:.3f}")
```

## Step 8: Visualize Acquisition Functions

```python
# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: GP Model
axes[0,0].plot(x_plot, y_plot, 'b-', linewidth=2, alpha=0.7, label='True function')
axes[0,0].plot(X_candidates, mean_pred, 'g-', linewidth=2, label='GP mean')
axes[0,0].fill_between(X_candidates.flatten(), 
                      mean_pred - 2*std_pred, mean_pred + 2*std_pred,
                      alpha=0.3, color='green', label='GP ±2σ')
axes[0,0].scatter(X_train, y_train, c='red', s=80, zorder=5, 
                 edgecolors='black', label='Training data')
axes[0,0].set_title('Gaussian Process Model')
axes[0,0].set_xlabel('x')
axes[0,0].set_ylabel('f(x)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Expected Improvement
axes[0,1].plot(X_candidates, ei_values, 'purple', linewidth=2)
axes[0,1].axvline(x=ei_point[0][0], color='red', linestyle='--',
                 label=f'EI max (x={ei_point[0][0]:.3f})')
axes[0,1].set_title('Expected Improvement')
axes[0,1].set_xlabel('x')
axes[0,1].set_ylabel('EI(x)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Upper Confidence Bound
axes[1,0].plot(X_candidates, ucb_values, 'orange', linewidth=2)
axes[1,0].axvline(x=ucb_point[0][0], color='red', linestyle='--',
                 label=f'UCB max (x={ucb_point[0][0]:.3f})')
axes[1,0].set_title('Upper Confidence Bound')
axes[1,0].set_xlabel('x')
axes[1,0].set_ylabel('UCB(x)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Probability of Improvement
axes[1,1].plot(X_candidates, poi_values, 'brown', linewidth=2)
axes[1,1].axvline(x=poi_point[0][0], color='red', linestyle='--',
                 label=f'PoI max (x={poi_point[0][0]:.3f})')
axes[1,1].set_title('Probability of Improvement')
axes[1,1].set_xlabel('x')
axes[1,1].set_ylabel('PoI(x)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 9: Simulate Next Experiment

```python
# Simulate conducting the experiment recommended by EI
next_x = ei_point[0][0]
true_value = objective_function(next_x)
measured_value = true_value + noise_level * np.random.randn()

print(f"\n🔬 Next Experiment:")
print(f"Recommended point: x = {next_x:.3f}")
print(f"True value: f({next_x:.3f}) = {true_value:.4f}")
print(f"Measured value (with noise): {measured_value:.4f}")
print(f"Current best: {y_train_series.min():.4f}")

if measured_value < y_train_series.min():
    improvement = y_train_series.min() - measured_value
    print(f"Improvement found! Δf = {improvement:.4f}")
else:
    print("No improvement this iteration (normal in optimization)")
```

## Step 10: Complete Optimization Loop

```python
def run_optimization_loop(n_iterations=10):
    """Run a complete optimization loop."""
    
    # Start with initial data
    X_current = X_train_df.copy()
    y_current = y_train_series.copy()
    
    history = {
        'iteration': [],
        'x_new': [],
        'y_new': [],
        'best_so_far': []
    }
    
    for i in range(n_iterations):
        print(f"\n--- Iteration {i+1} ---")
        
        # Fit model
        model = optimizer.fit(
            data_matrix=X_current,
            Measured_response=y_current,
            virtual_samples=X_candidates_df,
            min_search=True,
            Normalize=True
        )
        
        # Get next point using EI
        ei_values, ei_point = model.EI()
        next_x = ei_point[0][0]
        
        # Evaluate function (simulate experiment)
        true_y = objective_function(next_x)
        measured_y = true_y + noise_level * np.random.randn()
        
        # Update data
        new_row = pd.DataFrame({'x': [next_x]})
        X_current = pd.concat([X_current, new_row], ignore_index=True)
        y_current = pd.concat([y_current, pd.Series([measured_y])], ignore_index=True)
        
        # Track progress
        current_best = y_current.min()
        history['iteration'].append(i+1)
        history['x_new'].append(next_x)
        history['y_new'].append(measured_y)
        history['best_so_far'].append(current_best)
        
        print(f"New point: x = {next_x:.3f}, f(x) = {measured_y:.4f}")
        print(f"Best so far: {current_best:.4f}")
    
    return history, X_current, y_current

# Run optimization
history, X_final, y_final = run_optimization_loop(n_iterations=8)

print(f"\n Final Results:")
print(f"Best found: f({X_final.iloc[y_final.argmin()]['x']:.3f}) = {y_final.min():.4f}")
print(f"True minimum: f(1.750) = {objective_function(1.75):.4f}")
print(f"Error: {abs(y_final.min() - objective_function(1.75)):.4f}")
```

## Step 11: Analyze Results

```python
# Plot optimization progress
plt.figure(figsize=(12, 5))

# Convergence plot
plt.subplot(1, 2, 1)
plt.plot(history['iteration'], history['best_so_far'], 'bo-', linewidth=2, markersize=8)
plt.axhline(y=objective_function(1.75), color='red', linestyle='--',
           label=f'True minimum ({objective_function(1.75):.4f})')
plt.xlabel('Iteration')
plt.ylabel('Best Value Found')
plt.title('Optimization Convergence')
plt.legend()
plt.grid(True, alpha=0.3)

# Sampling locations
plt.subplot(1, 2, 2)
plt.plot(x_plot, y_plot, 'b-', linewidth=2, alpha=0.7, label='True function')
plt.scatter(X_train, y_train, c='red', s=100, marker='o', 
           edgecolors='black', label='Initial points', zorder=5)
plt.scatter(history['x_new'], history['y_new'], c='green', s=100, marker='s', 
           edgecolors='black', label='BO points', zorder=5)
plt.axvline(x=1.75, color='red', linestyle='--', alpha=0.7, label='True minimum')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Sampling Locations')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Key Takeaways

**What we learned:**

1. **Bayesian optimization is iterative** - Each new point improves our model
2. **Acquisition functions balance exploration vs exploitation** - EI found a good balance
3. **Uncertainty guides sampling** - Points with high uncertainty are explored
4. **Few evaluations needed** - We found the minimum with just a few experiments

**Next steps:**

- Try different acquisition functions: {doc}`acquisition_functions`
- Explore materials applications: {doc}`materials_discovery`
- Learn about multi-objective optimization: {doc}`multibgolearn`
- Try the GUI interface: {doc}`bgoface`

```{tip}
This simple example demonstrates the core Bgolearn workflow. The same principles apply to complex materials discovery problems with multiple objectives and constraints!
```
