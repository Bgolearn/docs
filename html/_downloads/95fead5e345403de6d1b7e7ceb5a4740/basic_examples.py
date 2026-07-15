#!/usr/bin/env python3
"""
Basic Examples for Bgolearn

This script demonstrates fundamental usage patterns of Bgolearn
for Bayesian optimization tasks.

Author: Bgolearn Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bgolearn as bgo
from bgolearn.visualization import BgolearnVisualizer

def example_1_simple_optimization():
    """
    Example 1: Simple 1D function optimization
    
    This example demonstrates the basic workflow:
    1. Create synthetic data
    2. Fit Bgolearn model
    3. Use acquisition functions
    4. Visualize results
    """
    print("=" * 60)
    print("Example 1: Simple 1D Function Optimization")
    print("=" * 60)
    
    # Create 1D test function (negative quadratic with noise)
    np.random.seed(42)
    
    def test_function(x):
        """Simple test function with global minimum at x=2."""
        return (x - 2)**2 + 0.1 * np.sin(10 * x)
    
    # Generate training data
    X_train_1d = np.random.uniform(-1, 5, 10).reshape(-1, 1)
    y_train_1d = np.array([test_function(x[0]) for x in X_train_1d])
    y_train_1d += 0.1 * np.random.randn(len(y_train_1d))  # Add noise
    
    # Generate candidate points
    X_candidates_1d = np.linspace(-1, 5, 100).reshape(-1, 1)
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_1d, columns=['x'])
    y_train_series = pd.Series(y_train_1d)
    X_candidates_df = pd.DataFrame(X_candidates_1d, columns=['x'])
    
    print(f"Training data: {len(X_train_df)} points")
    print(f"Candidate points: {len(X_candidates_df)} points")
    print(f"Current best: {y_train_series.min():.3f}")
    
    # Fit Bgolearn model
    optimizer = bgo.Bgolearn()
    model = optimizer.fit(
        data_matrix=X_train_df,
        Measured_response=y_train_series,
        virtual_samples=X_candidates_df,
        min_search=True,  # We want to minimize
        CV_test=3,
        Normalize=True
    )
    
    print("Model fitted successfully!")
    
    # Try different acquisition functions
    print("\nAcquisition function results:")
    
    # Expected Improvement
    ei_values, ei_point = model.EI()
    print(f"EI recommends: x = {ei_point[0]:.3f}")
    
    # Upper Confidence Bound
    ucb_values, ucb_point = model.UCB(alpha=2.0)
    print(f"UCB recommends: x = {ucb_point[0]:.3f}")
    
    # Probability of Improvement
    poi_values, poi_point = model.PoI(tao=0.01)
    print(f"PoI recommends: x = {poi_point[0]:.3f}")
    
    # Visualize results
    visualizer = BgolearnVisualizer(figsize=(15, 5))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: True function and training data
    x_true = np.linspace(-1, 5, 200)
    y_true = [test_function(x) for x in x_true]
    
    axes[0].plot(x_true, y_true, 'b-', label='True function', linewidth=2)
    axes[0].scatter(X_train_1d, y_train_1d, c='red', s=80, zorder=5, label='Training data')
    axes[0].scatter(ei_point[0], test_function(ei_point[0]), c='green', s=150, 
                   marker='*', zorder=5, label='EI recommendation')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('True Function and Training Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Model predictions
    axes[1].plot(x_true, y_true, 'b-', label='True function', linewidth=2)
    axes[1].plot(X_candidates_1d, model.virtual_samples_mean, 'g-', 
                label='GP mean', linewidth=2)
    axes[1].fill_between(X_candidates_1d.flatten(), 
                        model.virtual_samples_mean - 2*model.virtual_samples_std,
                        model.virtual_samples_mean + 2*model.virtual_samples_std,
                        alpha=0.3, color='green', label='GP uncertainty (±2σ)')
    axes[1].scatter(X_train_1d, y_train_1d, c='red', s=80, zorder=5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('f(x)')
    axes[1].set_title('Gaussian Process Model')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Acquisition function
    axes[2].plot(X_candidates_1d, ei_values, 'purple', linewidth=2, label='Expected Improvement')
    axes[2].axvline(x=ei_point[0], color='green', linestyle='--', 
                   label=f'EI maximum (x={ei_point[0]:.3f})')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('EI(x)')
    axes[2].set_title('Expected Improvement')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_1_simple_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, X_train_df, y_train_series, X_candidates_df


def example_2_2d_optimization():
    """
    Example 2: 2D function optimization
    
    Demonstrates optimization of a 2D function with visualization
    of the acquisition function landscape.
    """
    print("\n" + "=" * 60)
    print("Example 2: 2D Function Optimization")
    print("=" * 60)
    
    # Create 2D test function (Branin function variant)
    def branin_2d(x1, x2):
        """Modified Branin function."""
        a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)
        return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s
    
    # Generate training data
    np.random.seed(42)
    n_train = 15
    X_train_2d = np.random.uniform([-5, 0], [10, 15], (n_train, 2))
    y_train_2d = np.array([branin_2d(x[0], x[1]) for x in X_train_2d])
    y_train_2d += 0.5 * np.random.randn(n_train)  # Add noise
    
    # Generate candidate grid
    x1_range = np.linspace(-5, 10, 30)
    x2_range = np.linspace(0, 15, 30)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_candidates_2d = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_2d, columns=['x1', 'x2'])
    y_train_series = pd.Series(y_train_2d)
    X_candidates_df = pd.DataFrame(X_candidates_2d, columns=['x1', 'x2'])
    
    print(f"Training data: {len(X_train_df)} points")
    print(f"Candidate points: {len(X_candidates_df)} points")
    print(f"Current best: {y_train_series.min():.3f}")
    
    # Fit model
    optimizer = bgo.Bgolearn()
    model = optimizer.fit(
        data_matrix=X_train_df,
        Measured_response=y_train_series,
        virtual_samples=X_candidates_df,
        min_search=True,
        CV_test=5,
        Normalize=True
    )
    
    print("Model fitted successfully!")
    
    # Get acquisition function values
    ei_values, ei_point = model.EI()
    ucb_values, ucb_point = model.UCB(alpha=2.0)
    
    print(f"EI recommends: x1 = {ei_point[0]:.3f}, x2 = {ei_point[1]:.3f}")
    print(f"UCB recommends: x1 = {ucb_point[0]:.3f}, x2 = {ucb_point[1]:.3f}")
    
    # Visualize 2D results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # True function
    y_true_grid = np.array([branin_2d(x[0], x[1]) for x in X_candidates_2d])
    y_true_reshaped = y_true_grid.reshape(30, 30)
    
    im1 = axes[0,0].contourf(X1, X2, y_true_reshaped, levels=20, cmap='viridis', alpha=0.8)
    axes[0,0].contour(X1, X2, y_true_reshaped, levels=10, colors='white', alpha=0.5, linewidths=0.5)
    axes[0,0].scatter(X_train_2d[:, 0], X_train_2d[:, 1], c='white', s=80, edgecolors='black')
    axes[0,0].set_title('True Function')
    axes[0,0].set_xlabel('x1')
    axes[0,0].set_ylabel('x2')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Model predictions
    pred_reshaped = model.virtual_samples_mean.reshape(30, 30)
    im2 = axes[0,1].contourf(X1, X2, pred_reshaped, levels=20, cmap='viridis', alpha=0.8)
    axes[0,1].contour(X1, X2, pred_reshaped, levels=10, colors='white', alpha=0.5, linewidths=0.5)
    axes[0,1].scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_2d, cmap='plasma', 
                     s=80, edgecolors='white')
    axes[0,1].set_title('GP Predictions')
    axes[0,1].set_xlabel('x1')
    axes[0,1].set_ylabel('x2')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Expected Improvement
    ei_reshaped = ei_values.reshape(30, 30)
    im3 = axes[1,0].contourf(X1, X2, ei_reshaped, levels=20, cmap='plasma', alpha=0.8)
    axes[1,0].contour(X1, X2, ei_reshaped, levels=10, colors='white', alpha=0.5, linewidths=0.5)
    axes[1,0].scatter(X_train_2d[:, 0], X_train_2d[:, 1], c='white', s=60, edgecolors='black')
    axes[1,0].scatter(ei_point[0], ei_point[1], c='red', s=200, marker='*', 
                     edgecolors='white', linewidth=2)
    axes[1,0].set_title('Expected Improvement')
    axes[1,0].set_xlabel('x1')
    axes[1,0].set_ylabel('x2')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Upper Confidence Bound
    ucb_reshaped = ucb_values.reshape(30, 30)
    im4 = axes[1,1].contourf(X1, X2, ucb_reshaped, levels=20, cmap='coolwarm', alpha=0.8)
    axes[1,1].contour(X1, X2, ucb_reshaped, levels=10, colors='black', alpha=0.5, linewidths=0.5)
    axes[1,1].scatter(X_train_2d[:, 0], X_train_2d[:, 1], c='white', s=60, edgecolors='black')
    axes[1,1].scatter(ucb_point[0], ucb_point[1], c='red', s=200, marker='*', 
                     edgecolors='white', linewidth=2)
    axes[1,1].set_title('Upper Confidence Bound')
    axes[1,1].set_xlabel('x1')
    axes[1,1].set_ylabel('x2')
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('example_2_2d_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, X_train_df, y_train_series, X_candidates_df


def example_3_batch_optimization():
    """
    Example 3: Batch optimization for parallel experiments
    
    Demonstrates how to select multiple points for parallel evaluation.
    """
    print("\n" + "=" * 60)
    print("Example 3: Batch Optimization")
    print("=" * 60)
    
    # Use the 2D function from previous example
    def branin_2d(x1, x2):
        a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)
        return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s
    
    # Generate training data
    np.random.seed(42)
    n_train = 12
    X_train_2d = np.random.uniform([-5, 0], [10, 15], (n_train, 2))
    y_train_2d = np.array([branin_2d(x[0], x[1]) for x in X_train_2d])
    y_train_2d += 0.3 * np.random.randn(n_train)
    
    # Generate candidates
    x1_range = np.linspace(-5, 10, 25)
    x2_range = np.linspace(0, 15, 25)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_candidates_2d = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_2d, columns=['x1', 'x2'])
    y_train_series = pd.Series(y_train_2d)
    X_candidates_df = pd.DataFrame(X_candidates_2d, columns=['x1', 'x2'])
    
    # Fit model
    optimizer = bgo.Bgolearn()
    model = optimizer.fit(
        data_matrix=X_train_df,
        Measured_response=y_train_series,
        virtual_samples=X_candidates_df,
        min_search=True,
        Normalize=True
    )
    
    print("Model fitted for batch optimization")
    
    # Single point optimization for comparison
    ei_values, ei_single = model.EI()
    print(f"Single EI point: x1 = {ei_single[0]:.3f}, x2 = {ei_single[1]:.3f}")
    
    # Batch optimization
    batch_sizes = [3, 5, 7]
    batch_results = {}
    
    for q in batch_sizes:
        print(f"\nBatch size q = {q}:")
        
        # Batch Expected Improvement
        batch_ei = model.batch_EI(q=q, mc_samples=500)
        batch_indices, batch_points = batch_ei
        
        print(f"  Batch EI selected {len(batch_points)} points:")
        for i, point in enumerate(batch_points):
            print(f"    Point {i+1}: x1 = {point[0]:.3f}, x2 = {point[1]:.3f}")
        
        # Batch UCB
        batch_ucb = model.batch_UCB(q=q, beta=2.0)
        batch_indices_ucb, batch_points_ucb = batch_ucb
        
        print(f"  Batch UCB selected {len(batch_points_ucb)} points:")
        for i, point in enumerate(batch_points_ucb):
            print(f"    Point {i+1}: x1 = {point[0]:.3f}, x2 = {point[1]:.3f}")
        
        batch_results[q] = {
            'ei_points': batch_points,
            'ucb_points': batch_points_ucb
        }
    
    # Visualize batch selections
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot EI landscape for reference
    ei_reshaped = ei_values.reshape(25, 25)
    
    for i, q in enumerate(batch_sizes):
        # Batch EI
        im = axes[0, i].contourf(X1, X2, ei_reshaped, levels=15, cmap='viridis', alpha=0.8)
        axes[0, i].scatter(X_train_2d[:, 0], X_train_2d[:, 1], c='white', s=60, 
                          edgecolors='black', label='Training')
        
        # Plot batch EI points
        ei_points = batch_results[q]['ei_points']
        if ei_points:
            ei_x1 = [p[0] for p in ei_points]
            ei_x2 = [p[1] for p in ei_points]
            axes[0, i].scatter(ei_x1, ei_x2, c='red', s=120, marker='*', 
                              edgecolors='white', linewidth=2, label=f'Batch EI (q={q})')
        
        axes[0, i].set_title(f'Batch EI (q={q})')
        axes[0, i].set_xlabel('x1')
        axes[0, i].set_ylabel('x2')
        axes[0, i].legend()
        
        # Batch UCB
        ucb_values, _ = model.UCB(alpha=2.0)
        ucb_reshaped = ucb_values.reshape(25, 25)
        
        im2 = axes[1, i].contourf(X1, X2, ucb_reshaped, levels=15, cmap='plasma', alpha=0.8)
        axes[1, i].scatter(X_train_2d[:, 0], X_train_2d[:, 1], c='white', s=60, 
                          edgecolors='black', label='Training')
        
        # Plot batch UCB points
        ucb_points = batch_results[q]['ucb_points']
        if ucb_points:
            ucb_x1 = [p[0] for p in ucb_points]
            ucb_x2 = [p[1] for p in ucb_points]
            axes[1, i].scatter(ucb_x1, ucb_x2, c='red', s=120, marker='*', 
                              edgecolors='white', linewidth=2, label=f'Batch UCB (q={q})')
        
        axes[1, i].set_title(f'Batch UCB (q={q})')
        axes[1, i].set_xlabel('x1')
        axes[1, i].set_ylabel('x2')
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig('example_3_batch_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, batch_results


def example_4_acquisition_comparison():
    """
    Example 4: Comprehensive acquisition function comparison
    
    Compares different acquisition functions on the same problem.
    """
    print("\n" + "=" * 60)
    print("Example 4: Acquisition Function Comparison")
    print("=" * 60)
    
    # Create a more complex 1D function
    def complex_1d(x):
        """Complex 1D function with multiple local minima."""
        return (np.sin(3*x) + 0.3*np.cos(9*x) + 0.5*np.sin(7*x) + 
                0.1*(x-2)**2 + 0.05*x**3)
    
    # Generate training data
    np.random.seed(42)
    X_train_1d = np.array([-2, -1, 0, 1, 3, 4]).reshape(-1, 1)
    y_train_1d = np.array([complex_1d(x[0]) for x in X_train_1d])
    y_train_1d += 0.1 * np.random.randn(len(y_train_1d))
    
    # Generate candidates
    X_candidates_1d = np.linspace(-3, 5, 200).reshape(-1, 1)
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_1d, columns=['x'])
    y_train_series = pd.Series(y_train_1d)
    X_candidates_df = pd.DataFrame(X_candidates_1d, columns=['x'])
    
    # Fit model
    optimizer = bgo.Bgolearn()
    model = optimizer.fit(
        data_matrix=X_train_df,
        Measured_response=y_train_series,
        virtual_samples=X_candidates_df,
        min_search=True,
        Normalize=True
    )
    
    print("Model fitted for acquisition function comparison")
    
    # Compare different acquisition functions
    acquisition_functions = {
        'EI': lambda: model.EI(),
        'UCB (α=1.0)': lambda: model.UCB(alpha=1.0),
        'UCB (α=2.0)': lambda: model.UCB(alpha=2.0),
        'UCB (α=3.0)': lambda: model.UCB(alpha=3.0),
        'PoI (τ=0.0)': lambda: model.PoI(tao=0.0),
        'PoI (τ=0.1)': lambda: model.PoI(tao=0.1),
        'AEI': lambda: model.Augmented_EI(),
        'EQI (β=0.5)': lambda: model.EQI(beta=0.5)
    }
    
    results = {}
    for name, func in acquisition_functions.items():
        try:
            values, point = func()
            results[name] = {
                'values': values,
                'point': point[0],
                'max_value': values.max()
            }
            print(f"{name:15s}: recommends x = {point[0]:.3f}, max acq = {values.max():.4f}")
        except Exception as e:
            print(f"{name:15s}: Failed ({e})")
    
    # Visualize comparison
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.ravel()
    
    # True function and GP model
    x_true = np.linspace(-3, 5, 300)
    y_true = [complex_1d(x) for x in x_true]
    
    axes[0].plot(x_true, y_true, 'b-', label='True function', linewidth=2)
    axes[0].plot(X_candidates_1d, model.virtual_samples_mean, 'g-', 
                label='GP mean', linewidth=2)
    axes[0].fill_between(X_candidates_1d.flatten(), 
                        model.virtual_samples_mean - 2*model.virtual_samples_std,
                        model.virtual_samples_mean + 2*model.virtual_samples_std,
                        alpha=0.3, color='green', label='GP ±2σ')
    axes[0].scatter(X_train_1d, y_train_1d, c='red', s=80, zorder=5, label='Training data')
    axes[0].set_title('True Function and GP Model')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot each acquisition function
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (name, result) in enumerate(results.items()):
        if i < 8:  # We have 8 subplot slots remaining
            ax = axes[i + 1]
            ax.plot(X_candidates_1d, result['values'], color=colors[i], linewidth=2)
            ax.axvline(x=result['point'], color=colors[i], linestyle='--', alpha=0.7,
                      label=f'Max at x={result["point"]:.3f}')
            ax.set_title(f'{name}')
            ax.set_xlabel('x')
            ax.set_ylabel('Acquisition Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_4_acquisition_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, results


def example_5_iterative_optimization():
    """
    Example 5: Iterative optimization loop
    
    Demonstrates how to use Bgolearn in an iterative optimization loop,
    simulating a real experimental workflow.
    """
    print("\n" + "=" * 60)
    print("Example 5: Iterative Optimization Loop")
    print("=" * 60)
    
    # Define test function
    def objective_function(x1, x2):
        """Objective function to optimize."""
        return -(x1**2 + x2**2) + 0.1*np.sin(10*x1) + 0.1*np.cos(10*x2)
    
    # Initialize with small dataset
    np.random.seed(42)
    X_initial = np.random.uniform([-2, -2], [2, 2], (5, 2))
    y_initial = np.array([objective_function(x[0], x[1]) for x in X_initial])
    y_initial += 0.05 * np.random.randn(len(y_initial))  # Add noise
    
    # Create candidate space
    x1_range = np.linspace(-2, 2, 20)
    x2_range = np.linspace(-2, 2, 20)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_candidates = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Iterative optimization
    current_X = X_initial.copy()
    current_y = y_initial.copy()
    optimization_history = []
    
    n_iterations = 8
    
    print(f"Starting iterative optimization with {len(current_X)} initial points")
    print(f"Initial best: {current_y.max():.4f}")
    
    for iteration in range(n_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Convert to DataFrames
        X_df = pd.DataFrame(current_X, columns=['x1', 'x2'])
        y_series = pd.Series(current_y)
        X_candidates_df = pd.DataFrame(X_candidates, columns=['x1', 'x2'])
        
        # Fit model
        optimizer = bgo.Bgolearn()
        model = optimizer.fit(
            data_matrix=X_df,
            Measured_response=y_series,
            virtual_samples=X_candidates_df,
            min_search=False,  # Maximize
            Normalize=True
        )
        
        # Select next point using EI
        ei_values, next_point = model.EI()
        
        # Simulate experiment (evaluate true function)
        true_value = objective_function(next_point[0], next_point[1])
        measured_value = true_value + 0.05 * np.random.randn()  # Add noise
        
        # Update dataset
        current_X = np.vstack([current_X, next_point])
        current_y = np.append(current_y, measured_value)
        
        # Track progress
        current_best = current_y.max()
        optimization_history.append(current_best)
        
        print(f"Selected point: x1 = {next_point[0]:.3f}, x2 = {next_point[1]:.3f}")
        print(f"Measured value: {measured_value:.4f}")
        print(f"Current best: {current_best:.4f}")
        print(f"Total experiments: {len(current_X)}")
    
    # Final results
    best_idx = np.argmax(current_y)
    best_point = current_X[best_idx]
    best_value = current_y[best_idx]
    
    print(f"\n=== Final Results ===")
    print(f"Best point found: x1 = {best_point[0]:.3f}, x2 = {best_point[1]:.3f}")
    print(f"Best value: {best_value:.4f}")
    print(f"True optimum: 0.0 at (0, 0)")
    print(f"Total experiments: {len(current_X)}")
    
    # Visualize optimization progress
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Convergence plot
    axes[0,0].plot(range(len(optimization_history)), optimization_history, 'o-', 
                  linewidth=2, markersize=8)
    axes[0,0].axhline(y=0, color='red', linestyle='--', label='True optimum')
    axes[0,0].set_xlabel('Iteration')
    axes[0,0].set_ylabel('Best Objective Value')
    axes[0,0].set_title('Optimization Convergence')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # True function
    y_true_grid = np.array([objective_function(x[0], x[1]) for x in X_candidates])
    y_true_reshaped = y_true_grid.reshape(20, 20)
    
    im1 = axes[0,1].contourf(X1, X2, y_true_reshaped, levels=20, cmap='viridis', alpha=0.8)
    axes[0,1].scatter(current_X[:5, 0], current_X[:5, 1], c='white', s=80, 
                     edgecolors='black', label='Initial points')
    axes[0,1].scatter(current_X[5:, 0], current_X[5:, 1], c='red', s=60, 
                     label='BO selected points')
    axes[0,1].scatter(best_point[0], best_point[1], c='yellow', s=200, marker='*', 
                     edgecolors='black', linewidth=2, label='Best found')
    axes[0,1].set_title('True Function and Optimization Path')
    axes[0,1].legend()
    plt.colorbar(im1, ax=axes[0,1])
    
    # All experimental results
    axes[1,0].scatter(range(len(current_y)), current_y, c=range(len(current_y)), 
                     cmap='viridis', s=60, alpha=0.7)
    axes[1,0].axhline(y=best_value, color='red', linestyle='--', 
                     label=f'Best: {best_value:.4f}')
    axes[1,0].set_xlabel('Experiment Number')
    axes[1,0].set_ylabel('Measured Value')
    axes[1,0].set_title('All Experimental Results')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Improvement per iteration
    improvements = [optimization_history[0]]  # First iteration
    for i in range(1, len(optimization_history)):
        improvement = max(0, optimization_history[i] - optimization_history[i-1])
        improvements.append(improvement)
    
    axes[1,1].bar(range(len(improvements)), improvements, alpha=0.7)
    axes[1,1].set_xlabel('Iteration')
    axes[1,1].set_ylabel('Improvement')
    axes[1,1].set_title('Improvement per Iteration')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_5_iterative_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return current_X, current_y, optimization_history


def main():
    """Run all basic examples."""
    print("Running Bgolearn Basic Examples")
    print("=" * 80)
    
    # Run all examples
    try:
        model1, X1, y1, candidates1 = example_1_simple_optimization()
        model2, X2, y2, candidates2 = example_2_2d_optimization()
        model3, batch_results = example_3_batch_optimization()
        model4, acq_results = example_4_acquisition_comparison()
        final_X, final_y, history = example_5_iterative_optimization()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("Generated files:")
        print("- example_1_simple_optimization.png")
        print("- example_2_2d_optimization.png")
        print("- example_3_batch_optimization.png")
        print("- example_4_acquisition_comparison.png")
        print("- example_5_iterative_optimization.png")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
