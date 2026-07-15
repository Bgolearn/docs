# Visualization Guide

## 🎨 Introduction

Bgolearn provides comprehensive visualization tools to understand your optimization process, analyze acquisition functions, and communicate results effectively. This guide covers all visualization capabilities with practical examples.

## 📊 Basic Visualization

### Setting Up

```python
from Bgolearn import BGOsampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create sample optimization data
def create_optimization_example():
    """Create a complete optimization example for visualization."""
    np.random.seed(42)
    
    # Generate 2D test function (Branin function variant)
    def branin_function(x1, x2):
        a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)
        return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s
    
    # Training data
    n_train = 20
    X_train = np.random.uniform([-5, 0], [10, 15], (n_train, 2))
    y_train = np.array([branin_function(x[0], x[1]) for x in X_train])
    y_train += 0.5 * np.random.randn(n_train)  # Add noise
    
    # Candidate grid
    x1_range = np.linspace(-5, 10, 50)
    x2_range = np.linspace(0, 15, 50)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_candidates = np.column_stack([X1.ravel(), X2.ravel()])
    
    # True function values for comparison
    y_true = np.array([branin_function(x[0], x[1]) for x in X_candidates])
    
    return (pd.DataFrame(X_train, columns=['x1', 'x2']),
            pd.Series(y_train),
            pd.DataFrame(X_candidates, columns=['x1', 'x2']),
            y_true, X1, X2)

# Create data
X_train, y_train, X_candidates, y_true, X1, X2 = create_optimization_example()

print(f"Training data: {X_train.shape}")
print(f"Candidates: {X_candidates.shape}")
print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")

# Fit Bgolearn model
optimizer = BGOsampling.Bgolearn()
model = optimizer.fit(
    data_matrix=X_train,
    Measured_response=y_train,
    virtual_samples=X_candidates,
    min_search=True,  # Minimize the Branin function
    CV_test=5,
    Normalize=True
)

print("Model fitted successfully!")
```

### Initialize Visualizer

```python
# Create visualizer with custom settings
visualizer = BgolearnVisualizer(figsize=(12, 8), dpi=100)

# Note: Visualization features may vary by Bgolearn version
# Check available visualization methods in your Bgolearn installation
```

## 📈 Optimization History Visualization

### Basic Convergence Plot

```python
# Simulate optimization history
def simulate_optimization_history(n_iterations=15):
    """Simulate a realistic optimization history."""
    np.random.seed(42)
    
    # Start with current best
    current_best = y_train.min()
    history = [current_best]
    
    # Simulate improvement over iterations
    for i in range(1, n_iterations):
        # Diminishing returns with some noise
        improvement = np.random.exponential(0.3) * np.exp(-i/8)
        current_best = max(current_best - improvement, 0.5)  # Global minimum around 0.5
        history.append(current_best)
    
    return history

# Generate history
optimization_history = simulate_optimization_history()

# Plot basic convergence
fig = visualizer.plot_optimization_history(
    y_history=optimization_history,
    y_true_optimum=0.398,  # Known Branin minimum
    title="Bayesian Optimization Convergence",
    save_path="convergence_basic.png"
)
plt.show()
```

### Advanced Convergence Analysis

```python
# Multiple algorithm comparison
def create_algorithm_comparison():
    """Compare different acquisition functions."""
    base_history = optimization_history
    
    algorithms = {
        'Expected Improvement': base_history,
        'Upper Confidence Bound': [h + 0.1 + 0.05*np.random.randn() for h in base_history],
        'Probability of Improvement': [h + 0.2 + 0.03*np.random.randn() for h in base_history],
        'Random Search': [base_history[0] - 0.1*i + 0.2*np.random.randn() 
                         for i in range(len(base_history))]
    }
    
    return algorithms

# Create comparison
algorithm_histories = create_algorithm_comparison()

# Plot comparison
fig = visualizer.plot_convergence_comparison(
    histories=algorithm_histories,
    title="Acquisition Function Performance Comparison",
    save_path="algorithm_comparison.png"
)
plt.show()

# Add statistical analysis
print("Final Performance Summary:")
print("-" * 40)
for name, history in algorithm_histories.items():
    final_value = history[-1]
    improvement = history[0] - final_value
    print(f"{name:25s}: Final={final_value:.3f}, Improvement={improvement:.3f}")
```

## 🎯 Acquisition Function Visualization

### 1D Acquisition Functions

```python
def visualize_1d_acquisition():
    """Demonstrate 1D acquisition function visualization."""
    
    # Create 1D problem
    np.random.seed(42)
    x_1d = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_1d = -(x_1d.flatten() - 0.5)**2 + 0.2 * np.random.randn(100)
    
    # Select training points
    train_indices = [10, 25, 40, 60, 80]
    X_train_1d = x_1d[train_indices]
    y_train_1d = y_1d[train_indices]
    
    # Fit 1D model
    optimizer_1d = BGOsampling.Bgolearn()
    model_1d = optimizer_1d.fit(
        pd.DataFrame(X_train_1d, columns=['x']),
        pd.Series(y_train_1d),
        pd.DataFrame(x_1d, columns=['x']),
        min_search=False
    )
    
    # Get acquisition values
    ei_values, next_point = model_1d.EI()
    
    # Plot 1D acquisition function
    fig = visualizer.plot_acquisition_function_1d(
        X_candidates=x_1d,
        acquisition_values=ei_values,
        X_train=X_train_1d,
        y_train=y_train_1d,
        mean_pred=model_1d.virtual_samples_mean,
        std_pred=model_1d.virtual_samples_std,
        next_point=next_point[0] if len(next_point) > 0 else None,
        title="1D Expected Improvement Visualization",
        save_path="acquisition_1d.png"
    )
    plt.show()

# Run 1D visualization
visualize_1d_acquisition()
```

### 2D Acquisition Functions

```python
# Get acquisition function values for 2D visualization
ei_values, next_point = model.EI()
ucb_values, _ = model.UCB(alpha=2.0)

# Reshape for 2D plotting
grid_size = int(np.sqrt(len(ei_values)))
ei_grid = ei_values[:grid_size**2].reshape(grid_size, grid_size)
ucb_grid = ucb_values[:grid_size**2].reshape(grid_size, grid_size)

# Plot 2D acquisition functions
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Expected Improvement
im1 = axes[0,0].contourf(X1, X2, ei_grid, levels=20, cmap='viridis', alpha=0.8)
axes[0,0].contour(X1, X2, ei_grid, levels=10, colors='white', alpha=0.5, linewidths=0.5)
axes[0,0].scatter(X_train['x1'], X_train['x2'], c=y_train, cmap='plasma', 
                 s=100, edgecolors='white', linewidth=2)
if len(next_point) > 0:
    axes[0,0].scatter(next_point[0], next_point[1], c='red', s=200, marker='*', 
                     edgecolors='white', linewidth=2)
axes[0,0].set_title('Expected Improvement')
axes[0,0].set_xlabel('x1')
axes[0,0].set_ylabel('x2')
plt.colorbar(im1, ax=axes[0,0])

# Upper Confidence Bound
im2 = axes[0,1].contourf(X1, X2, ucb_grid, levels=20, cmap='plasma', alpha=0.8)
axes[0,1].contour(X1, X2, ucb_grid, levels=10, colors='white', alpha=0.5, linewidths=0.5)
axes[0,1].scatter(X_train['x1'], X_train['x2'], c=y_train, cmap='viridis', 
                 s=100, edgecolors='white', linewidth=2)
axes[0,1].set_title('Upper Confidence Bound')
axes[0,1].set_xlabel('x1')
axes[0,1].set_ylabel('x2')
plt.colorbar(im2, ax=axes[0,1])

# True function
y_true_grid = y_true[:grid_size**2].reshape(grid_size, grid_size)
im3 = axes[1,0].contourf(X1, X2, y_true_grid, levels=20, cmap='coolwarm', alpha=0.8)
axes[1,0].contour(X1, X2, y_true_grid, levels=10, colors='black', alpha=0.5, linewidths=0.5)
axes[1,0].scatter(X_train['x1'], X_train['x2'], c='white', s=100, 
                 edgecolors='black', linewidth=2)
axes[1,0].set_title('True Function')
axes[1,0].set_xlabel('x1')
axes[1,0].set_ylabel('x2')
plt.colorbar(im3, ax=axes[1,0])

# Model predictions
pred_grid = model.virtual_samples_mean[:grid_size**2].reshape(grid_size, grid_size)
im4 = axes[1,1].contourf(X1, X2, pred_grid, levels=20, cmap='coolwarm', alpha=0.8)
axes[1,1].contour(X1, X2, pred_grid, levels=10, colors='black', alpha=0.5, linewidths=0.5)
axes[1,1].scatter(X_train['x1'], X_train['x2'], c=y_train, cmap='plasma', 
                 s=100, edgecolors='white', linewidth=2)
axes[1,1].set_title('Model Predictions')
axes[1,1].set_xlabel('x1')
axes[1,1].set_ylabel('x2')
plt.colorbar(im4, ax=axes[1,1])

plt.tight_layout()
plt.savefig('acquisition_2d_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Acquisition Function Evolution

```python
def visualize_acquisition_evolution():
    """Show how acquisition function changes over iterations."""
    
    # Simulate acquisition function evolution
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # Different stages of optimization
    stages = [
        ("Initial (5 points)", 5),
        ("Early (10 points)", 10), 
        ("Mid (15 points)", 15),
        ("Late (20 points)", 20),
        ("Final (25 points)", 25),
        ("Converged (30 points)", 30)
    ]
    
    for i, (stage_name, n_points) in enumerate(stages):
        # Use subset of training data to simulate progression
        if n_points <= len(X_train):
            X_subset = X_train.iloc[:n_points]
            y_subset = y_train.iloc[:n_points]
        else:
            X_subset = X_train
            y_subset = y_train
        
        # Fit model with subset
        optimizer_subset = BGOsampling.Bgolearn()
        model_subset = optimizer_subset.fit(
            X_subset, y_subset, X_candidates, min_search=True, Normalize=True
        )
        
        # Get acquisition values
        ei_subset, _ = model_subset.EI()
        ei_subset_grid = ei_subset[:grid_size**2].reshape(grid_size, grid_size)
        
        # Plot
        im = axes[i].contourf(X1, X2, ei_subset_grid, levels=15, cmap='viridis', alpha=0.8)
        axes[i].scatter(X_subset['x1'], X_subset['x2'], c='white', s=60, 
                       edgecolors='black', linewidth=1)
        axes[i].set_title(f'{stage_name}')
        axes[i].set_xlabel('x1')
        axes[i].set_ylabel('x2')
    
    plt.tight_layout()
    plt.savefig('acquisition_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run evolution visualization
visualize_acquisition_evolution()
```

## 🔄 Batch Optimization Visualization

```python
# Visualize batch selection
def visualize_batch_selection():
    """Visualize batch optimization results."""
    
    # Get different batch selections
    batch_ei = model.batch_EI(q=5, mc_samples=500)
    batch_ucb = model.batch_UCB(q=5, beta=2.0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Single point EI
    ei_vals, single_point = model.EI()
    ei_grid = ei_vals[:grid_size**2].reshape(grid_size, grid_size)
    
    im1 = axes[0].contourf(X1, X2, ei_grid, levels=20, cmap='viridis', alpha=0.8)
    axes[0].scatter(X_train['x1'], X_train['x2'], c=y_train, cmap='plasma', 
                   s=100, edgecolors='white', linewidth=2)
    if len(single_point) > 0:
        axes[0].scatter(single_point[0], single_point[1], c='red', s=200, marker='*',
                       edgecolors='white', linewidth=2, label='Single EI')
    axes[0].set_title('Single Point EI')
    axes[0].legend()
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Batch EI
    im2 = axes[1].contourf(X1, X2, ei_grid, levels=20, cmap='viridis', alpha=0.8)
    axes[1].scatter(X_train['x1'], X_train['x2'], c=y_train, cmap='plasma', 
                   s=100, edgecolors='white', linewidth=2)
    
    batch_ei_points = batch_ei[1]
    if len(batch_ei_points) > 0:
        batch_x1 = [p[0] for p in batch_ei_points]
        batch_x2 = [p[1] for p in batch_ei_points]
        axes[1].scatter(batch_x1, batch_x2, c='red', s=150, marker='*',
                       edgecolors='white', linewidth=2, label='Batch EI')
    axes[1].set_title('Batch EI (q=5)')
    axes[1].legend()
    plt.colorbar(im2, ax=axes[1])
    
    # Plot 3: Batch UCB
    ucb_vals, _ = model.UCB(alpha=2.0)
    ucb_grid = ucb_vals[:grid_size**2].reshape(grid_size, grid_size)
    
    im3 = axes[2].contourf(X1, X2, ucb_grid, levels=20, cmap='plasma', alpha=0.8)
    axes[2].scatter(X_train['x1'], X_train['x2'], c=y_train, cmap='viridis', 
                   s=100, edgecolors='white', linewidth=2)
    
    batch_ucb_points = batch_ucb[1]
    if len(batch_ucb_points) > 0:
        batch_ucb_x1 = [p[0] for p in batch_ucb_points]
        batch_ucb_x2 = [p[1] for p in batch_ucb_points]
        axes[2].scatter(batch_ucb_x1, batch_ucb_x2, c='red', s=150, marker='*',
                       edgecolors='white', linewidth=2, label='Batch UCB')
    axes[2].set_title('Batch UCB (q=5)')
    axes[2].legend()
    plt.colorbar(im3, ax=axes[2])
    
    for ax in axes:
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
    
    plt.tight_layout()
    plt.savefig('batch_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run batch visualization
visualize_batch_selection()
```

## 📊 Model Analysis Visualization

### Uncertainty Visualization

```python
def visualize_model_uncertainty():
    """Visualize model uncertainty and predictions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Model mean predictions
    mean_grid = model.virtual_samples_mean[:grid_size**2].reshape(grid_size, grid_size)
    im1 = axes[0,0].contourf(X1, X2, mean_grid, levels=20, cmap='coolwarm', alpha=0.8)
    axes[0,0].scatter(X_train['x1'], X_train['x2'], c=y_train, cmap='plasma', 
                     s=100, edgecolors='white', linewidth=2)
    axes[0,0].set_title('Model Mean Predictions')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Model uncertainty (standard deviation)
    std_grid = model.virtual_samples_std[:grid_size**2].reshape(grid_size, grid_size)
    im2 = axes[0,1].contourf(X1, X2, std_grid, levels=20, cmap='Reds', alpha=0.8)
    axes[0,1].scatter(X_train['x1'], X_train['x2'], c='white', s=100, 
                     edgecolors='black', linewidth=2)
    axes[0,1].set_title('Model Uncertainty (Std Dev)')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Prediction error (if we have true values)
    error_grid = np.abs(mean_grid - y_true_grid)
    im3 = axes[1,0].contourf(X1, X2, error_grid, levels=20, cmap='Oranges', alpha=0.8)
    axes[1,0].scatter(X_train['x1'], X_train['x2'], c='white', s=100, 
                     edgecolors='black', linewidth=2)
    axes[1,0].set_title('Prediction Error')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Uncertainty vs Error correlation
    axes[1,1].scatter(std_grid.ravel(), error_grid.ravel(), alpha=0.5, s=20)
    axes[1,1].set_xlabel('Model Uncertainty')
    axes[1,1].set_ylabel('Prediction Error')
    axes[1,1].set_title('Uncertainty vs Error')
    
    # Add correlation coefficient
    correlation = np.corrcoef(std_grid.ravel(), error_grid.ravel())[0,1]
    axes[1,1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=axes[1,1].transAxes, bbox=dict(boxstyle="round", facecolor='white'))
    
    for ax in axes.flat:
        if hasattr(ax, 'set_xlabel'):
            ax.set_xlabel('x1')
        if hasattr(ax, 'set_ylabel') and ax != axes[1,1]:
            ax.set_ylabel('x2')
    
    plt.tight_layout()
    plt.savefig('model_uncertainty.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run uncertainty visualization
visualize_model_uncertainty()
```

### Cross-Validation Visualization

```python
def visualize_cv_results():
    """Visualize cross-validation results."""
    
    # Note: In practice, you'd extract CV results from the model
    # Here we simulate for demonstration
    np.random.seed(42)
    
    # Simulate CV predictions
    cv_predictions = y_train + 0.3 * np.random.randn(len(y_train))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Predicted vs Actual
    axes[0].scatter(y_train, cv_predictions, alpha=0.7, s=80)
    min_val, max_val = min(y_train.min(), cv_predictions.min()), max(y_train.max(), cv_predictions.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Cross-Validation: Predicted vs Actual')
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_train, cv_predictions)
    axes[0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0].transAxes,
                bbox=dict(boxstyle="round", facecolor='white'))
    
    # Residuals
    residuals = cv_predictions - y_train
    axes[1].scatter(cv_predictions, residuals, alpha=0.7, s=80)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    
    # Residual histogram
    axes[2].hist(residuals, bins=10, alpha=0.7, edgecolor='black')
    axes[2].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig('cv_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("Cross-Validation Statistics:")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.3f}")
    print(f"MAE: {np.mean(np.abs(residuals)):.3f}")

# Run CV visualization
visualize_cv_results()
```

## 🎯 Materials Science Specific Visualizations

### Composition-Property Relationships

```python
def visualize_materials_properties():
    """Visualize materials-specific relationships."""
    
    # Create materials-like data
    np.random.seed(42)
    n_materials = 50
    
    # Composition data (ternary system)
    comp_A = np.random.uniform(0.1, 0.8, n_materials)
    comp_B = np.random.uniform(0.1, 0.9 - comp_A, n_materials)
    comp_C = 1 - comp_A - comp_B
    
    # Processing conditions
    temperature = np.random.uniform(800, 1200, n_materials)
    pressure = np.random.uniform(1, 10, n_materials)
    
    # Properties (simulate complex relationships)
    strength = (100 * comp_A * comp_B + 50 * comp_C + 
               0.1 * temperature - 2 * pressure + 
               10 * np.random.randn(n_materials))
    
    ductility = (80 * comp_C + 30 * comp_A - 0.05 * temperature + 
                pressure + 5 * np.random.randn(n_materials))
    
    # Create comprehensive materials visualization
    fig = plt.figure(figsize=(20, 15))
    
    # Ternary composition plot (simplified as 2D)
    ax1 = plt.subplot(3, 3, 1)
    scatter1 = ax1.scatter(comp_A, comp_B, c=strength, cmap='viridis', s=80, alpha=0.7)
    ax1.set_xlabel('Composition A')
    ax1.set_ylabel('Composition B')
    ax1.set_title('Strength vs Composition')
    plt.colorbar(scatter1, ax=ax1, label='Strength')
    
    # Temperature-Pressure map
    ax2 = plt.subplot(3, 3, 2)
    scatter2 = ax2.scatter(temperature, pressure, c=strength, cmap='plasma', s=80, alpha=0.7)
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Pressure (GPa)')
    ax2.set_title('Strength vs Processing Conditions')
    plt.colorbar(scatter2, ax=ax2, label='Strength')
    
    # Property correlation
    ax3 = plt.subplot(3, 3, 3)
    scatter3 = ax3.scatter(strength, ductility, c=comp_A, cmap='coolwarm', s=80, alpha=0.7)
    ax3.set_xlabel('Strength')
    ax3.set_ylabel('Ductility')
    ax3.set_title('Property Trade-off')
    plt.colorbar(scatter3, ax=ax3, label='Comp A')
    
    # Composition effects
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(comp_A, strength, alpha=0.7, label='Comp A', s=60)
    ax4.scatter(comp_B, strength, alpha=0.7, label='Comp B', s=60)
    ax4.scatter(comp_C, strength, alpha=0.7, label='Comp C', s=60)
    ax4.set_xlabel('Composition Fraction')
    ax4.set_ylabel('Strength')
    ax4.set_title('Individual Composition Effects')
    ax4.legend()
    
    # Processing parameter effects
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(temperature, strength, alpha=0.7, s=60)
    ax5.set_xlabel('Temperature (K)')
    ax5.set_ylabel('Strength')
    ax5.set_title('Temperature Effect')
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(pressure, strength, alpha=0.7, s=60)
    ax6.set_xlabel('Pressure (GPa)')
    ax6.set_ylabel('Strength')
    ax6.set_title('Pressure Effect')
    
    # Property distributions
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(strength, bins=15, alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Strength')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Strength Distribution')
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(ductility, bins=15, alpha=0.7, edgecolor='black', color='orange')
    ax8.set_xlabel('Ductility')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Ductility Distribution')
    
    # Pareto front (if multi-objective)
    ax9 = plt.subplot(3, 3, 9)
    ax9.scatter(strength, ductility, alpha=0.7, s=60)
    
    # Find approximate Pareto front
    pareto_indices = []
    for i in range(len(strength)):
        is_pareto = True
        for j in range(len(strength)):
            if i != j and strength[j] >= strength[i] and ductility[j] >= ductility[i]:
                if strength[j] > strength[i] or ductility[j] > ductility[i]:
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    
    if pareto_indices:
        pareto_strength = [strength[i] for i in pareto_indices]
        pareto_ductility = [ductility[i] for i in pareto_indices]
        ax9.scatter(pareto_strength, pareto_ductility, c='red', s=100, 
                   marker='*', label='Pareto Front')
        ax9.legend()
    
    ax9.set_xlabel('Strength')
    ax9.set_ylabel('Ductility')
    ax9.set_title('Multi-Objective Pareto Front')
    
    plt.tight_layout()
    plt.savefig('materials_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# Run materials visualization
visualize_materials_properties()
```

## 🎮 Interactive Visualizations

### Plotly Integration

```python
def create_interactive_dashboard():
    """Create interactive Plotly dashboard."""
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        
        # Create interactive dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Optimization History', 'Acquisition Function',
                           'Model Predictions', 'Uncertainty'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Optimization history
        fig.add_trace(
            go.Scatter(x=list(range(len(optimization_history))),
                      y=optimization_history,
                      mode='lines+markers',
                      name='Best Value',
                      line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # Acquisition function (sample points)
        sample_indices = np.random.choice(len(ei_values), 200, replace=False)
        fig.add_trace(
            go.Scatter(x=X_candidates.iloc[sample_indices]['x1'],
                      y=X_candidates.iloc[sample_indices]['x2'],
                      mode='markers',
                      marker=dict(size=8, color=ei_values[sample_indices], 
                                colorscale='Viridis', showscale=True),
                      name='EI Values'),
            row=1, col=2
        )
        
        # Model predictions
        fig.add_trace(
            go.Scatter(x=X_candidates.iloc[sample_indices]['x1'],
                      y=X_candidates.iloc[sample_indices]['x2'],
                      mode='markers',
                      marker=dict(size=8, color=model.virtual_samples_mean[sample_indices], 
                                colorscale='Plasma', showscale=True),
                      name='Predictions'),
            row=2, col=1
        )
        
        # Uncertainty
        fig.add_trace(
            go.Scatter(x=X_candidates.iloc[sample_indices]['x1'],
                      y=X_candidates.iloc[sample_indices]['x2'],
                      mode='markers',
                      marker=dict(size=8, color=model.virtual_samples_std[sample_indices], 
                                colorscale='Reds', showscale=True),
                      name='Uncertainty'),
            row=2, col=2
        )
        
        # Add training data to relevant plots
        for row, col in [(1, 2), (2, 1), (2, 2)]:
            fig.add_trace(
                go.Scatter(x=X_train['x1'], y=X_train['x2'],
                          mode='markers',
                          marker=dict(size=12, color='white', 
                                    line=dict(color='black', width=2)),
                          name='Training Data',
                          showlegend=(row==1 and col==2)),
                row=row, col=col
            )
        
        fig.update_layout(height=800, title_text="Bgolearn Interactive Dashboard")
        fig.show()
        
        # Save as HTML
        fig.write_html("bgolearn_dashboard.html")
        print("Interactive dashboard saved as 'bgolearn_dashboard.html'")
        
    except ImportError:
        print("Plotly not available. Install with: pip install plotly")

# Create interactive dashboard
create_interactive_dashboard()
```

## 💡 Visualization Best Practices

### Custom Styling

```python
# Set up custom matplotlib style
plt.style.use('seaborn-v0_8')
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def setup_publication_style():
    """Set up publication-ready plot style."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

setup_publication_style()
```

### Saving High-Quality Figures

```python
def save_publication_figure(fig, filename, formats=['png', 'pdf', 'svg']):
    """Save figure in multiple high-quality formats."""
    
    for fmt in formats:
        fig.savefig(f"{filename}.{fmt}", 
                   format=fmt, 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
    
    print(f"Saved {filename} in formats: {', '.join(formats)}")

# Example usage
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(optimization_history, 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Iteration')
ax.set_ylabel('Best Objective Value')
ax.set_title('Optimization Convergence')
ax.grid(True, alpha=0.3)

save_publication_figure(fig, 'convergence_publication')
plt.show()
```

## 🎯 Quick Visualization Functions

```python
# Use the built-in quick visualization
quick_fig = quick_optimization_plot(
    model=model,
    X_train=X_train.values,
    y_train=y_train.values,
    X_candidates=X_candidates.values,
    acquisition_func='EI',
    save_path='quick_optimization.png'
)

# Integrated visualization from main class
optimizer.plot_optimization_progress(save_path='integrated_progress.png')

print("All visualizations completed!")
print("Generated files:")
print("- convergence_basic.png")
print("- algorithm_comparison.png")
print("- acquisition_1d.png")
print("- acquisition_2d_comparison.png")
print("- batch_comparison.png")
print("- model_uncertainty.png")
print("- materials_analysis.png")
print("- bgolearn_dashboard.html")
print("- quick_optimization.png")
print("- integrated_progress.png")

## 📚 Summary

This comprehensive visualization guide covers:

1. **Basic Plots**: Convergence, acquisition functions, model predictions
2. **Advanced Analysis**: Uncertainty, cross-validation, algorithm comparison
3. **Batch Optimization**: Multi-point selection visualization
4. **Materials Science**: Composition-property relationships, Pareto fronts
5. **Interactive Tools**: Plotly dashboards and dynamic visualizations
6. **Best Practices**: Publication-ready styling and high-quality exports

The visualization tools in Bgolearn help you:
- Understand optimization progress and convergence
- Analyze acquisition function behavior
- Validate model performance
- Communicate results effectively
- Make informed decisions about next experiments

For more examples and advanced techniques, see the [Examples](examples/) section.
```
