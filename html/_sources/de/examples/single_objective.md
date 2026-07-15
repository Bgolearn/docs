# Single-Objective Optimization Examples

```{note}
Diese Seite ist die deutsche Version des Bgolearn-Handbuchs.
```

```{note}
This page provides comprehensive examples of single-objective optimization using the original Bgolearn API.
```

## Example 1: Alloy Strength Optimization

Let's optimize the strength of an aluminum alloy by adjusting the composition of alloying elements.

### Problem Setup

We want to maximize the yield strength of an Al-Cu-Mg-Si alloy by optimizing the composition of Cu, Mg, and Si.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bgolearn import BGOsampling

# Initialize Bgolearn
opt = BGOsampling.Bgolearn()
```

### Prepare Training Data

```python
# Historical experimental data
# Features: Cu (%), Mg (%), Si (%)
# Target: Yield Strength (MPa)

# CRITICAL: Use pandas DataFrame (REQUIRED by Bgolearn)
data_matrix = pd.DataFrame({
    'Cu_content': [2.0, 3.5, 1.8, 4.2, 2.8, 3.2, 2.5, 3.8],  # Cu (%)
    'Mg_content': [1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.9, 1.0],  # Mg (%)
    'Si_content': [0.5, 0.7, 0.3, 0.8, 0.6, 0.4, 0.9, 0.5]   # Si (%)
})

# CRITICAL: Use pandas Series (REQUIRED by Bgolearn)
measured_response = pd.Series([250, 280, 240, 290, 265, 275, 255, 285],
                             name='Yield_Strength')

print(f"Training data: {len(data_matrix)} experiments")
print(f"Features: {list(data_matrix.columns)}")
print(f"Feature ranges:")
print(f"  Cu: {data_matrix['Cu_content'].min():.1f} - {data_matrix['Cu_content'].max():.1f}%")
print(f"  Mg: {data_matrix['Mg_content'].min():.1f} - {data_matrix['Mg_content'].max():.1f}%")
print(f"  Si: {data_matrix['Si_content'].min():.1f} - {data_matrix['Si_content'].max():.1f}%")
print(f"Strength range: {measured_response.min()} - {measured_response.max()} MPa")
```

### Define Virtual Space

```python
# Generate candidate compositions for optimization
np.random.seed(42)  # For reproducibility

# Create a grid of candidate compositions
cu_range = np.linspace(1.5, 4.5, 10)
mg_range = np.linspace(0.7, 1.6, 8)
si_range = np.linspace(0.2, 1.0, 9)

# Create all combinations
virtual_samples_list = []
for cu in cu_range:
    for mg in mg_range:
        for si in si_range:
            # Add composition constraint: total alloying elements < 7%
            if cu + mg + si < 7.0:
                virtual_samples_list.append([cu, mg, si])

# CRITICAL: Convert to pandas DataFrame (REQUIRED by Bgolearn)
virtual_samples = pd.DataFrame(virtual_samples_list,
                              columns=['Cu_content', 'Mg_content', 'Si_content'])
print(f"Virtual space: {len(virtual_samples)} candidate compositions")
print(f"Virtual samples columns: {list(virtual_samples.columns)}")
```

### Run Optimization

```python
# Optimize for maximum strength
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Mission='Regression',           # Continuous target
    Classifier='GaussianProcess',   # Surrogate model
    opt_num=1,                     # Single recommendation
    min_search=False,              # False = maximization
    CV_test=4,                    # 4-fold cross-validation
    Dynamic_W=False,               # Static weights
    Normalize=True,                # Normalize features
    seed=42                        # Random seed
)

print("Optimization completed!")
```

### Analyze Results

```python
# Get the recommended composition using EI
ei_values, recommended_points = model.EI()
recommended_composition = recommended_points[0]

print(f"\nRecommended next experiment:")
print(f"Composition: Cu={recommended_composition[0]:.2f}%, Mg={recommended_composition[1]:.2f}%, Si={recommended_composition[2]:.2f}%")
print(f"Total alloying content: {sum(recommended_composition):.2f}%")

# Get acquisition function values
print(f"Maximum EI value: {np.max(ei_values):.4f}")

# Get model predictions for all virtual samples
predicted_strengths = model.virtual_samples_mean
prediction_stds = model.virtual_samples_std

# Find the index of recommended point for prediction
recommended_idx = np.argmax(ei_values)
print(f"Predicted strength: {predicted_strengths[recommended_idx]:.1f} ± {prediction_stds[recommended_idx]:.1f} MPa")
```

### Visualize Results

```python
# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Training data scatter plot
ax1 = axes[0, 0]
scatter = ax1.scatter(data_matrix['Cu_content'], data_matrix['Mg_content'],
                     c=measured_response, s=100, cmap='viridis',
                     edgecolors='black', linewidth=1)
ax1.scatter(recommended_composition[0], recommended_composition[1],
           c='red', s=200, marker='*', edgecolors='black', linewidth=2,
           label='Recommended')
ax1.set_xlabel('Cu (%)')
ax1.set_ylabel('Mg (%)')
ax1.set_title('Training Data and Recommendation')
ax1.legend()
plt.colorbar(scatter, ax=ax1, label='Strength (MPa)')

# 2. Acquisition function landscape
ax2 = axes[0, 1]
# Create a grid for visualization
cu_grid = np.linspace(1.5, 4.5, 50)
mg_grid = np.linspace(0.7, 1.6, 40)
Cu_mesh, Mg_mesh = np.meshgrid(cu_grid, mg_grid)

# For visualization, fix Si at the recommended value
si_fixed = recommended_composition[2]
grid_points = np.column_stack([Cu_mesh.ravel(), Mg_mesh.ravel(), 
                              np.full(Cu_mesh.size, si_fixed)])

# Get acquisition values for grid (simplified)
# Note: This is a conceptual visualization
acq_grid = np.random.random(Cu_mesh.size)  # Placeholder
acq_mesh = acq_grid.reshape(Cu_mesh.shape)

contour = ax2.contourf(Cu_mesh, Mg_mesh, acq_mesh, levels=20, cmap='plasma')
ax2.scatter(recommended_composition[0], recommended_composition[1], 
           c='white', s=200, marker='*', edgecolors='black', linewidth=2)
ax2.set_xlabel('Cu (%)')
ax2.set_ylabel('Mg (%)')
ax2.set_title(f'Acquisition Function (Si={si_fixed:.2f}%)')
plt.colorbar(contour, ax=ax2, label='Acquisition Value')

# 3. Predicted vs Actual (Training)
ax3 = axes[1, 0]

# Compute predictions on training data by fitting with virtual_samples=data_matrix
model_train = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=data_matrix,
    Mission='Regression',
    Classifier='GaussianProcess',
    opt_num=1,
    min_search=False,
    CV_test=4,
    Normalize=True,
    seed=42
)
train_predictions = model_train.virtual_samples_mean

ax3.scatter(measured_response, train_predictions, s=100, alpha=0.7)
ax3.plot([measured_response.min(), measured_response.max()],
         [measured_response.min(), measured_response.max()], 'r--', lw=2)
ax3.set_xlabel('Actual Strength (MPa)')
ax3.set_ylabel('Predicted Strength (MPa)')
ax3.set_title('Predicted vs Actual (Training)')
ax3.grid(True, alpha=0.3)

# Calculate R²
r2_score = np.corrcoef(measured_response, train_predictions)[0, 1]**2
ax3.text(0.05, 0.95, f'R² = {r2_score:.3f}', transform=ax3.transAxes,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 4. Feature importance (if available)
ax4 = axes[1, 1]
feature_names = ['Cu', 'Mg', 'Si']
# Placeholder feature importance
feature_importance = np.array([0.45, 0.35, 0.20])
bars = ax4.bar(feature_names, feature_importance, color=['skyblue', 'lightgreen', 'salmon'])
ax4.set_ylabel('Feature Importance')
ax4.set_title('Feature Importance Analysis')
ax4.set_ylim(0, 0.6)

# Add value labels on bars
for bar, importance in zip(bars, feature_importance):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{importance:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

## Example 2: Processing Parameter Optimization

Optimize heat treatment parameters to maximize hardness.

### Problem Setup

```python
# Heat treatment optimization
# Features: Temperature (°C), Time (hours), Cooling Rate (°C/min)
# Target: Hardness (HV)

# CRITICAL: Use pandas DataFrame (REQUIRED by Bgolearn)
processing_data = pd.DataFrame({
    'Temperature': [450, 500, 550, 480, 520, 470, 530, 490],  # °C
    'Time': [2, 4, 6, 3, 5, 2.5, 4.5, 3.5],                 # hours
    'Cooling_Rate': [10, 20, 15, 25, 12, 18, 22, 14]        # °C/min
})

# CRITICAL: Use pandas Series (REQUIRED by Bgolearn)
hardness_values = pd.Series([180, 220, 250, 200, 235, 190, 245, 210],
                           name='Hardness_HV')

# Virtual processing conditions
temp_range = np.linspace(440, 560, 25)
time_range = np.linspace(1.5, 6.5, 20)
cooling_range = np.linspace(8, 28, 15)

virtual_processing_list = []
for temp in temp_range:
    for time in time_range:
        for cooling in cooling_range:
            # Add practical constraints
            if 1.5 <= time <= 6.5 and 8 <= cooling <= 28:
                virtual_processing_list.append([temp, time, cooling])

# CRITICAL: Convert to pandas DataFrame (REQUIRED by Bgolearn)
virtual_processing = pd.DataFrame(virtual_processing_list,
                                 columns=['Temperature', 'Time', 'Cooling_Rate'])
print(f"Virtual processing space: {len(virtual_processing)} conditions")
print(f"Processing columns: {list(processing_data.columns)}")
```

### Optimization with Different Models

```python
# Compare different surrogate models
models_to_test = ['GaussianProcess', 'RandomForest', 'SVR', 'MLP']
results = {}

for model_name in models_to_test:
    print(f"\nTesting {model_name}...")
    
    model = opt.fit(
        data_matrix=processing_data,
        Measured_response=hardness_values,
        virtual_samples=virtual_processing,
        Mission='Regression',
        Classifier=model_name,
        opt_num=1,
        min_search=False,  # Maximize hardness
        CV_test='LOOCV',  # Leave-one-out cross-validation
        Normalize=True
    )
    
    ei_values, recommended_points = model.EI()
    recommended_conditions = recommended_points[0]
    recommended_idx = np.argmax(ei_values)
    predicted_hardness = model.virtual_samples_mean[recommended_idx]

    results[model_name] = {
        'conditions': recommended_conditions,
        'predicted_hardness': predicted_hardness,
        'ei_max': np.max(ei_values)
    }
    
    print(f"Recommended: T={recommended_conditions[0]:.0f}°C, "
          f"t={recommended_conditions[1]:.1f}h, CR={recommended_conditions[2]:.0f}°C/min")
    print(f"Predicted hardness: {predicted_hardness:.1f} HV")

# Compare results
print("\nModel Comparison:")
print("-" * 60)
for model_name, result in results.items():
    conditions = result['conditions']
    print(f"{model_name:15s}: T={conditions[0]:5.0f}°C, t={conditions[1]:4.1f}h, "
          f"CR={conditions[2]:4.0f}°C/min, HV={result['predicted_hardness']:5.1f}")
```

## Example 3: Multi-Response Optimization

Optimize for multiple responses using weighted objectives.

### Problem Setup

```python
# Optimize alloy for both strength and ductility
# We'll use a weighted approach to combine objectives

# CRITICAL: Use pandas DataFrame (REQUIRED by Bgolearn)
alloy_data = pd.DataFrame({
    'Cu_content': [2.0, 3.5, 1.8, 4.2, 2.8, 3.2, 2.5, 3.8, 2.2, 3.6],  # Cu (%)
    'Mg_content': [1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.9, 1.0, 1.4, 0.7],  # Mg (%)
    'Si_content': [0.5, 0.7, 0.3, 0.8, 0.6, 0.4, 0.9, 0.5, 0.7, 0.6]   # Si (%)
})

# Multiple responses
strength_values = np.array([250, 280, 240, 290, 265, 275, 255, 285, 245, 275])
ductility_values = np.array([15, 12, 18, 10, 14, 13, 16, 11, 17, 12])

# Create weighted objective
# Weight: 60% strength, 40% ductility
strength_weight = 0.6
ductility_weight = 0.4

# Normalize responses to [0, 1] scale
strength_norm = (strength_values - strength_values.min()) / (strength_values.max() - strength_values.min())
ductility_norm = (ductility_values - ductility_values.min()) / (ductility_values.max() - ductility_values.min())

# Combined objective
combined_objective_values = strength_weight * strength_norm + ductility_weight * ductility_norm

# CRITICAL: Use pandas Series (REQUIRED by Bgolearn)
combined_objective = pd.Series(combined_objective_values, name='Combined_Objective')

print(f"Training data: {len(alloy_data)} experiments")
print(f"Features: {list(alloy_data.columns)}")
print(f"Strength range: {strength_values.min()} - {strength_values.max()} MPa")
print(f"Ductility range: {ductility_values.min()} - {ductility_values.max()} %")
print(f"Combined objective range: {combined_objective.min():.3f} - {combined_objective.max():.3f}")
```


## Example 5: Multi-Objective Optimization with MultiBgolearn

Here's how to properly perform multi-objective optimization using the MultiBgolearn package with EHVI:

```python
# Multi-objective optimization using MultiBgolearn with EHVI
print("=== Multi-Objective Optimization with MultiBgolearn EHVI ===")

import numpy as np
import pandas as pd
from MultiBgolearn import bgo

# Note: MultiBgolearn uses a different API that works with CSV files

# Create multi-objective training data
alloy_compositions = pd.DataFrame({
    'Cu_content': [2.0, 3.5, 1.8, 4.2, 2.8, 3.2, 2.5, 3.8, 2.2, 3.6],
    'Mg_content': [1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.9, 1.0, 1.4, 0.7],
    'Si_content': [0.5, 0.7, 0.3, 0.8, 0.6, 0.4, 0.9, 0.5, 0.7, 0.6]
})

# Multiple objectives
strength_values = np.array([250, 280, 240, 290, 265, 275, 255, 285, 245, 275])
ductility_values = np.array([15, 12, 18, 10, 14, 13, 16, 11, 17, 12])
corrosion_resistance = np.array([8.5, 7.2, 9.1, 6.8, 8.0, 7.8, 8.3, 7.0, 9.0, 7.5])

print(f"Training data: {len(alloy_compositions)} experiments")
print(f"Objectives: Strength (MPa), Ductility (%), Corrosion Resistance (score)")

# Method 1: Weighted Sum Approach
print("\n--- Method 1: Weighted Sum Approach ---")

# Define weights for objectives (must sum to 1.0)
w_strength = 0.5      # 50% weight on strength
w_ductility = 0.3     # 30% weight on ductility
w_corrosion = 0.2     # 20% weight on corrosion resistance

# Normalize objectives to [0, 1] scale
strength_norm = (strength_values - strength_values.min()) / (strength_values.max() - strength_values.min())
ductility_norm = (ductility_values - ductility_values.min()) / (ductility_values.max() - ductility_values.min())
corrosion_norm = (corrosion_resistance - corrosion_resistance.min()) / (corrosion_resistance.max() - corrosion_resistance.min())

# Create weighted objective
weighted_objective = (w_strength * strength_norm +
                     w_ductility * ductility_norm +
                     w_corrosion * corrosion_norm)

# Convert to pandas Series
multi_objective = pd.Series(weighted_objective, name='Weighted_Objective')

# Create virtual samples
virtual_candidates = []
for cu in np.linspace(1.5, 4.5, 8):
    for mg in np.linspace(0.7, 1.6, 6):
        for si in np.linspace(0.2, 1.0, 5):
            if cu + mg + si < 7.0:  # Composition constraint
                virtual_candidates.append([cu, mg, si])

virtual_samples = pd.DataFrame(virtual_candidates,
                              columns=['Cu_content', 'Mg_content', 'Si_content'])

print(f"Virtual space: {len(virtual_samples)} candidates")
print(f"Objective weights: Strength={w_strength}, Ductility={w_ductility}, Corrosion={w_corrosion}")

# Optimize weighted objective
model_weighted = opt.fit(
    data_matrix=alloy_compositions,
    Measured_response=multi_objective,
    virtual_samples=virtual_samples,
    Mission='Regression',
    Classifier='GaussianProcess',
    opt_num=1,
    min_search=False,  # Maximize weighted objective
    CV_test=5,
    Normalize=True,
    seed=42
)

# Get recommendation
ei_values, recommended_points = model_weighted.EI()
best_composition = recommended_points[0]

print(f"\nWeighted Sum Results:")
print(f"Recommended composition: Cu={best_composition[0]:.2f}%, Mg={best_composition[1]:.2f}%, Si={best_composition[2]:.2f}%")

# Get prediction
best_idx = np.argmax(ei_values)
predicted_objective = model_weighted.virtual_samples_mean[best_idx]
print(f"Predicted weighted objective: {predicted_objective:.3f}")

# Method 2: Sequential Optimization
print("\n--- Method 2: Sequential Optimization ---")

# First optimize for strength
strength_series = pd.Series(strength_values, name='Strength')

model_strength = opt.fit(
    data_matrix=alloy_compositions,
    Measured_response=strength_series,
    virtual_samples=virtual_samples,
    Mission='Regression',
    Classifier='GaussianProcess',
    opt_num=1,
    min_search=False,  # Maximize strength
    CV_test=5,
    Normalize=True,
    seed=42
)

ei_strength, points_strength = model_strength.EI()
best_strength_comp = points_strength[0]

print(f"Best for strength: Cu={best_strength_comp[0]:.2f}%, Mg={best_strength_comp[1]:.2f}%, Si={best_strength_comp[2]:.2f}%")

# Then optimize for ductility
ductility_series = pd.Series(ductility_values, name='Ductility')

model_ductility = opt.fit(
    data_matrix=alloy_compositions,
    Measured_response=ductility_series,
    virtual_samples=virtual_samples,
    Mission='Regression',
    Classifier='GaussianProcess',
    opt_num=1,
    min_search=False,  # Maximize ductility
    CV_test=5,
    Normalize=True,
    seed=42
)

ei_ductility, points_ductility = model_ductility.EI()
best_ductility_comp = points_ductility[0]

print(f"Best for ductility: Cu={best_ductility_comp[0]:.2f}%, Mg={best_ductility_comp[1]:.2f}%, Si={best_ductility_comp[2]:.2f}%")

# Method 3: Pareto Front Analysis
print("\n--- Method 3: Pareto Front Analysis ---")

# Get predictions for all objectives
strength_pred = model_strength.virtual_samples_mean
ductility_pred = model_ductility.virtual_samples_mean

# Find Pareto optimal solutions
def is_pareto_optimal(costs, i):
    """Check if point i is Pareto optimal"""
    return not np.any(np.all(costs >= costs[i], axis=1) & np.any(costs > costs[i], axis=1))

# Combine objectives (negate for minimization)
objectives = np.column_stack([-strength_pred, -ductility_pred])
pareto_mask = [is_pareto_optimal(objectives, i) for i in range(len(objectives))]
pareto_indices = np.where(pareto_mask)[0]

print(f"Found {len(pareto_indices)} Pareto optimal solutions")

# Show top 3 Pareto solutions
if len(pareto_indices) >= 3:
    top_pareto = pareto_indices[:3]
    for i, idx in enumerate(top_pareto):
        comp = virtual_samples.iloc[idx]
        print(f"Pareto solution {i+1}: Cu={comp['Cu_content']:.2f}%, Mg={comp['Mg_content']:.2f}%, Si={comp['Si_content']:.2f}%")
        print(f"  Predicted strength: {strength_pred[idx]:.1f} MPa, ductility: {ductility_pred[idx]:.1f}%")

print("\n Multi-objective optimization completed using current Bgolearn API!")
```

### Key Points for Multi-Objective Optimization:

1. **Weighted Sum**: Combine objectives with user-defined weights
2. **Sequential Optimization**: Optimize each objective separately
3. **Pareto Analysis**: Find trade-off solutions
4. **Current API**: Uses standard `fit()` method with pandas data
5. **No Special Parameters**: Works with existing Bgolearn functionality

## Best Practices Summary

### Data Quality
1. **Sufficient data**: Aim for >5 samples per feature
2. **Representative sampling**: Cover the design space well
3. **Quality control**: Remove outliers and errors
4. **Feature scaling**: Normalize features with different units

### Model Selection
1. **Start simple**: Begin with Gaussian Process
2. **Cross-validate**: Use CV_test=10 or 'LOOCV' for validation
3. **Compare models**: Test multiple surrogate models
4. **Validate predictions**: Check model performance

### Optimization Strategy
1. **Define objectives clearly**: Maximize or minimize?
2. **Handle constraints**: Filter virtual space or use penalties
3. **Multiple recommendations**: Use opt_num > 1 for alternatives
4. **Iterative improvement**: Add new data and re-optimize

### Practical Considerations
1. **Experimental validation**: Always validate recommendations
2. **Uncertainty quantification**: Consider prediction uncertainty
3. **Domain knowledge**: Incorporate expert knowledge
4. **Resource constraints**: Balance exploration vs. exploitation

## Next Steps

- **Learn multi-objective optimization**: {doc}`../multibgolearn`
- **Understand acquisition functions**: {doc}`../acquisition_functions`
