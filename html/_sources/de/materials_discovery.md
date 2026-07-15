# Materials Discovery with Bgolearn

```{note}
Diese Seite ist die deutsche Version des Bgolearn-Handbuchs.
```

## Introduction

Bgolearn is specifically designed for materials discovery workflows, providing specialized tools and examples for accelerating materials research through Bayesian optimization. This guide demonstrates how to apply Bgolearn to real materials science problems.

## Materials Discovery Workflow

### Problem Definition

```python
from Bgolearn import BGOsampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define a materials discovery problem
class MaterialsOptimizationProblem:
    """
    Example: Optimizing alloy composition for maximum strength.
    
    This example demonstrates a typical materials discovery workflow
    where we want to find the optimal composition and processing
    conditions for a new alloy.
    """
    
    def __init__(self):
        self.feature_names = [
            'Al_content',      # Aluminum content (0-1)
            'Cu_content',      # Copper content (0-1) 
            'Mg_content',      # Magnesium content (0-1)
            'Temperature',     # Processing temperature (K)
            'Pressure',        # Processing pressure (GPa)
            'Time'            # Processing time (hours)
        ]
        
        self.target_properties = [
            'Tensile_Strength',  # MPa
            'Ductility',        # %
            'Corrosion_Resistance'  # Arbitrary units
        ]
    
    def generate_initial_data(self, n_samples=30):
        """Generate initial experimental data."""
        np.random.seed(42)
        
        # Composition constraints (sum to 1, with balance element)
        al_content = np.random.uniform(0.1, 0.7, n_samples)
        cu_content = np.random.uniform(0.05, 0.3, n_samples)
        mg_content = np.random.uniform(0.01, 0.2, n_samples)
        
        # Normalize compositions
        total_content = al_content + cu_content + mg_content
        al_content = al_content / total_content * 0.9  # Leave 10% for other elements
        cu_content = cu_content / total_content * 0.9
        mg_content = mg_content / total_content * 0.9
        
        # Processing conditions
        temperature = np.random.uniform(400, 800, n_samples)  # K
        pressure = np.random.uniform(0.1, 5.0, n_samples)    # GPa
        time = np.random.uniform(0.5, 24, n_samples)         # hours
        
        # Create feature matrix
        X = np.column_stack([al_content, cu_content, mg_content, 
                           temperature, pressure, time])
        
        # Simulate complex property relationships
        tensile_strength = self._calculate_tensile_strength(X)
        ductility = self._calculate_ductility(X)
        corrosion_resistance = self._calculate_corrosion_resistance(X)
        
        # Create DataFrames
        features_df = pd.DataFrame(X, columns=self.feature_names)
        properties_df = pd.DataFrame({
            'Tensile_Strength': tensile_strength,
            'Ductility': ductility,
            'Corrosion_Resistance': corrosion_resistance
        })
        
        return features_df, properties_df
    
    def _calculate_tensile_strength(self, X):
        """Simulate tensile strength calculation."""
        al, cu, mg, temp, pressure, time = X.T
        
        # Complex non-linear relationships
        base_strength = 200 + 300 * al + 400 * cu + 150 * mg
        temp_effect = 1 - 0.001 * (temp - 600)**2 / 600  # Optimal around 600K
        pressure_effect = 1 + 0.1 * np.log(pressure + 0.1)
        time_effect = 1 + 0.05 * np.log(time + 0.1)
        interaction = 50 * al * cu  # Al-Cu interaction
        
        strength = (base_strength * temp_effect * pressure_effect * 
                   time_effect + interaction + 20 * np.random.randn(len(al)))
        
        return np.maximum(strength, 100)  # Minimum strength
    
    def _calculate_ductility(self, X):
        """Simulate ductility calculation."""
        al, cu, mg, temp, pressure, time = X.T
        
        # Ductility often trades off with strength
        base_ductility = 15 + 10 * mg - 5 * cu  # Mg increases, Cu decreases ductility
        temp_effect = 1 + 0.002 * (temp - 500)  # Higher temp increases ductility
        pressure_effect = 1 - 0.05 * pressure   # Pressure decreases ductility
        
        ductility = (base_ductility * temp_effect * pressure_effect + 
                    2 * np.random.randn(len(al)))
        
        return np.maximum(ductility, 1)  # Minimum ductility
    
    def _calculate_corrosion_resistance(self, X):
        """Simulate corrosion resistance calculation."""
        al, cu, mg, temp, pressure, time = X.T
        
        # Al and Mg improve corrosion resistance, Cu decreases it
        base_resistance = 5 + 3 * al + 2 * mg - 1 * cu
        processing_effect = 1 + 0.001 * temp - 0.1 * pressure
        
        resistance = (base_resistance * processing_effect + 
                     0.5 * np.random.randn(len(al)))
        
        return np.maximum(resistance, 1)  # Minimum resistance

# Create materials problem
materials_problem = MaterialsOptimizationProblem()
X_initial, y_initial = materials_problem.generate_initial_data(n_samples=25)

# Create training data variables for consistency
X_train = X_initial.copy()
y_train = y_initial['Tensile_Strength'].copy()  # Focus on tensile strength

print("Initial experimental data:")
print(f"Features shape: {X_initial.shape}")
print(f"Properties shape: {y_initial.shape}")
print(f"Training data shape: {X_train.shape}")
print(f"Training target shape: {y_train.shape}")
print("\nFeature ranges:")
print(X_initial.describe())
print("\nProperty ranges:")
print(y_initial.describe())
```

### Candidate Space Generation

```python
def generate_materials_candidates(n_candidates=500):
    """
    Generate candidate materials for optimization.
    
    This function creates a diverse set of candidate compositions
    and processing conditions while respecting materials constraints.
    """
    np.random.seed(123)
    
    candidates = []
    
    for _ in range(n_candidates):
        # Generate compositions with constraints
        al = np.random.uniform(0.1, 0.8)
        cu = np.random.uniform(0.05, min(0.4, 0.9 - al))
        mg = np.random.uniform(0.01, min(0.3, 0.9 - al - cu))
        
        # Normalize to ensure realistic compositions
        total = al + cu + mg
        if total > 0.9:  # Leave room for other elements
            al = al / total * 0.9
            cu = cu / total * 0.9
            mg = mg / total * 0.9
        
        # Processing conditions
        temp = np.random.uniform(400, 800)
        pressure = np.random.uniform(0.1, 5.0)
        time = np.random.uniform(0.5, 24)
        
        candidates.append([al, cu, mg, temp, pressure, time])
    
    return pd.DataFrame(candidates, columns=materials_problem.feature_names)

# Generate candidate space
X_candidates = generate_materials_candidates(n_candidates=400)

print(f"Generated {len(X_candidates)} candidate materials")
print("\nCandidate composition ranges:")
print(X_candidates[['Al_content', 'Cu_content', 'Mg_content']].describe())

# Visualize candidate space
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Composition triangle (simplified as 2D)
axes[0,0].scatter(X_candidates['Al_content'], X_candidates['Cu_content'], 
                 c=X_candidates['Mg_content'], cmap='viridis', alpha=0.6)
axes[0,0].set_xlabel('Al Content')
axes[0,0].set_ylabel('Cu Content')
axes[0,0].set_title('Composition Space (colored by Mg)')

# Processing conditions
axes[0,1].scatter(X_candidates['Temperature'], X_candidates['Pressure'], 
                 alpha=0.6, c='blue')
axes[0,1].set_xlabel('Temperature (K)')
axes[0,1].set_ylabel('Pressure (GPa)')
axes[0,1].set_title('Processing Conditions')

# Time distribution
axes[0,2].hist(X_candidates['Time'], bins=20, alpha=0.7, edgecolor='black')
axes[0,2].set_xlabel('Processing Time (hours)')
axes[0,2].set_ylabel('Frequency')
axes[0,2].set_title('Time Distribution')

# Initial data properties
axes[1,0].scatter(y_initial['Tensile_Strength'], y_initial['Ductility'], 
                 c=y_initial['Corrosion_Resistance'], cmap='plasma', s=80)
axes[1,0].set_xlabel('Tensile Strength (MPa)')
axes[1,0].set_ylabel('Ductility (%)')
axes[1,0].set_title('Property Trade-offs (colored by Corrosion Resistance)')

# Property distributions
axes[1,1].hist(y_initial['Tensile_Strength'], bins=10, alpha=0.7, 
              edgecolor='black', label='Tensile Strength')
axes[1,1].set_xlabel('Tensile Strength (MPa)')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Strength Distribution')

axes[1,2].hist(y_initial['Ductility'], bins=10, alpha=0.7, 
              edgecolor='black', color='orange', label='Ductility')
axes[1,2].set_xlabel('Ductility (%)')
axes[1,2].set_ylabel('Frequency')
axes[1,2].set_title('Ductility Distribution')

plt.tight_layout()
plt.savefig('materials_candidate_space.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Single-Objective Optimization

```python
# Optimize for maximum tensile strength
print("=== Single-Objective Optimization: Tensile Strength ===")

optimizer_strength = BGOsampling.Bgolearn()
model_strength = optimizer_strength.fit(
    data_matrix=X_train,
    Measured_response=y_train,
    virtual_samples=X_candidates,
    min_search=False,  # Maximize strength
    CV_test=5,
    Normalize=True
)

print("Model fitted for tensile strength optimization")

# Try different acquisition functions
acquisition_results = {}

# Expected Improvement
ei_values, ei_point = model_strength.EI()
acquisition_results['EI'] = {
    'point': ei_point,
    'predicted_strength': model_strength.virtual_samples_mean[np.argmax(ei_values)]
}

# Upper Confidence Bound
ucb_values, ucb_point = model_strength.UCB(alpha=2.0)
acquisition_results['UCB'] = {
    'point': ucb_point,
    'predicted_strength': model_strength.virtual_samples_mean[np.argmax(ucb_values)]
}

# Probability of Improvement
poi_values, poi_point = model_strength.PoI(tao=0.01)
acquisition_results['PoI'] = {
    'point': poi_point,
    'predicted_strength': model_strength.virtual_samples_mean[np.argmax(poi_values)]
}

# Display results
print("\nAcquisition Function Recommendations:")
print("-" * 60)
for method, result in acquisition_results.items():
    point = result['point']
    pred_strength = result['predicted_strength']

    print(f"{method}:")
    print(f"  Al: {point[0][0]:.3f}, Cu: {point[0][1]:.3f}, Mg: {point[0][2]:.3f}")
    print(f"  Temp: {point[0][3]:.1f}K, Pressure: {point[0][4]:.2f}GPa, Time: {point[0][5]:.1f}h")
    print(f"  Predicted Strength: {pred_strength:.1f} MPa")
    print()

# Multiple recommendations for sequential experiments
print("=== Multiple Recommendations for Sequential Experiments ===")

# Get multiple recommendations using opt_num parameter
model_multi = opt.fit(
    data_matrix=X_train,
    Measured_response=y_train,
    virtual_samples=X_candidates,
    opt_num=5,  # Request 5 recommendations
    min_search=False,
    Normalize=True
)

# Get recommendations using Expected Improvement
ei_values, recommended_points = model_multi.EI()

# The model returns the best recommendation
# For multiple experiments, you can use the top EI values
top_indices = np.argsort(ei_values)[-5:][::-1]  # Top 5 indices

print(f"Top 5 recommended experiments:")
for i, idx in enumerate(top_indices):
    point = X_candidates.iloc[idx]
    pred_strength = model_multi.virtual_samples_mean[idx]
    ei_value = ei_values[idx]

    print(f"  Rank {i+1} (EI={ei_value:.4f}):")
    print(f"    Composition: Al={point[0]:.3f}, Cu={point[1]:.3f}, Mg={point[2]:.3f}")
    print(f"    Processing: T={point[3]:.1f}K, P={point[4]:.2f}GPa, t={point[5]:.1f}h")
    print(f"    Predicted Strength: {pred_strength:.1f} MPa")
    print()
```

### Multi-Objective Optimization

```python
# Multi-objective optimization: Strength vs Ductility
print("=== Multi-Objective Optimization: Strength vs Ductility ===")

# Fit separate models for each property using Bgolearn
print("Fitting model for ductility optimization...")
opt_ductility = BGOsampling.Bgolearn()
fitted_model_ductility = opt_ductility.fit(
    data_matrix=X_train,
    Measured_response=y_initial['Ductility'],
    virtual_samples=X_candidates,
    min_search=False,  # Maximize ductility
    CV_test=3,
    Normalize=True
)

print("Getting predictions from both models...")
# Get predictions for all candidates using Bgolearn's built-in functionality
strength_pred = model_strength.virtual_samples_mean
ductility_pred = fitted_model_ductility.virtual_samples_mean

# Create combined objective (weighted sum)
# User can adjust weights based on priorities
w_strength = 0.6  # Weight for strength
w_ductility = 0.4  # Weight for ductility

# Normalize objectives to [0, 1] scale
strength_norm = (strength_pred - strength_pred.min()) / (strength_pred.max() - strength_pred.min())
ductility_norm = (ductility_pred - ductility_pred.min()) / (ductility_pred.max() - ductility_pred.min())

combined_objective = w_strength * strength_norm + w_ductility * ductility_norm

# Find best multi-objective solution
best_idx = np.argmax(combined_objective)
best_point = X_candidates.iloc[best_idx]
    
print("Multi-Objective Optimization Results:")
print(f"Weights: Strength={w_strength}, Ductility={w_ductility}")
print(f"Best composition:")
print(f"  Al: {best_point['Al_content']:.3f}")
print(f"  Cu: {best_point['Cu_content']:.3f}")
print(f"  Mg: {best_point['Mg_content']:.3f}")
print(f"Processing conditions:")
print(f"  Temperature: {best_point['Temperature']:.1f} K")
print(f"  Pressure: {best_point['Pressure']:.2f} GPa")
print(f"  Time: {best_point['Time']:.1f} hours")
print(f"Predicted properties:")
print(f"  Tensile Strength: {strength_pred[best_idx]:.1f} MPa")
print(f"  Ductility: {ductility_pred[best_idx]:.1f} %")
    
# Find Pareto front approximation using Bgolearn predictions
pareto_indices = []
for i in range(len(strength_pred)):
    is_pareto = True
    for j in range(len(strength_pred)):
        if (strength_pred[j] >= strength_pred[i] and
            ductility_pred[j] >= ductility_pred[i] and
            (strength_pred[j] > strength_pred[i] or ductility_pred[j] > ductility_pred[i])):
            is_pareto = False
            break
    if is_pareto:
        pareto_indices.append(i)

print(f"\nFound {len(pareto_indices)} points on Pareto front")
    
# Visualize multi-objective results using matplotlib
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Pareto front
axes[0].scatter(strength_pred, ductility_pred, alpha=0.5, s=20, label='All candidates')
if pareto_indices:
    pareto_strength = strength_pred[pareto_indices]
    pareto_ductility = ductility_pred[pareto_indices]
    axes[0].scatter(pareto_strength, pareto_ductility, c='red', s=80,
                   marker='*', label='Pareto front')
axes[0].scatter(strength_pred[best_idx], ductility_pred[best_idx],
               c='blue', s=150, marker='D', label='Best weighted')
axes[0].set_xlabel('Predicted Tensile Strength (MPa)')
axes[0].set_ylabel('Predicted Ductility (%)')
axes[0].set_title('Multi-Objective Trade-off')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
    
# Combined objective landscape
scatter = axes[1].scatter(X_candidates['Al_content'], X_candidates['Cu_content'],
                         c=combined_objective, cmap='viridis', s=30, alpha=0.7)
axes[1].scatter(best_point['Al_content'], best_point['Cu_content'],
               c='red', s=150, marker='*', edgecolors='white', linewidth=2)
axes[1].set_xlabel('Al Content')
axes[1].set_ylabel('Cu Content')
axes[1].set_title('Combined Objective in Composition Space')
plt.colorbar(scatter, ax=axes[1], label='Combined Objective')

# Weight sensitivity analysis
weights = np.linspace(0, 1, 11)
best_strengths = []
best_ductilities = []

for w_s in weights:
    w_d = 1 - w_s
    combined = w_s * strength_norm + w_d * ductility_norm
    best_idx_w = np.argmax(combined)
    best_strengths.append(strength_pred[best_idx_w])
    best_ductilities.append(ductility_pred[best_idx_w])

axes[2].plot(weights, best_strengths, 'o-', label='Strength', linewidth=2)
axes[2].set_xlabel('Weight for Strength')
axes[2].set_ylabel('Predicted Tensile Strength (MPa)', color='blue')
axes[2].tick_params(axis='y', labelcolor='blue')

ax2_twin = axes[2].twinx()
ax2_twin.plot(weights, best_ductilities, 's-', color='orange', label='Ductility', linewidth=2)
ax2_twin.set_ylabel('Predicted Ductility (%)', color='orange')
ax2_twin.tick_params(axis='y', labelcolor='orange')

axes[2].set_title('Weight Sensitivity Analysis')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multi_objective_optimization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Multi-objective optimization completed using Bgolearn's built-in capabilities!")
```

### Complete Working Example

Here's a complete, working example that demonstrates proper data formatting for Bgolearn:

```python
# Materials Discovery with Bgolearn - Complete Working Example
print("=== Materials Discovery with Bgolearn ===")

import numpy as np
import pandas as pd
from Bgolearn import BGOsampling
import matplotlib.pyplot as plt

# Step 1: Create sample materials data with proper pandas format
print("Creating sample materials dataset...")

# Generate synthetic aluminum alloy data
np.random.seed(42)
n_samples = 25

# Composition features (Al, Cu, Mg content)
al_content = np.random.uniform(0.80, 0.95, n_samples)
cu_content = np.random.uniform(0.02, 0.15, n_samples)
mg_content = np.random.uniform(0.01, 0.08, n_samples)

# Processing parameters
temperature = np.random.uniform(400, 600, n_samples)  # °C
pressure = np.random.uniform(1, 5, n_samples)         # GPa
time = np.random.uniform(1, 10, n_samples)            # hours

# CRITICAL: Create DataFrame with column names (REQUIRED by Bgolearn)
data_matrix = pd.DataFrame({
    'Al_content': al_content,
    'Cu_content': cu_content,
    'Mg_content': mg_content,
    'Temperature': temperature,
    'Pressure': pressure,
    'Time': time
})

# Simulate tensile strength (target property)
strength = (
    500 * al_content +           # Al base strength
    800 * cu_content +           # Cu strengthening
    1200 * mg_content +          # Mg strengthening
    0.5 * temperature +          # Temperature effect
    20 * pressure +              # Pressure effect
    -5 * time +                  # Time degradation
    np.random.normal(0, 20, n_samples)  # Measurement noise
)

# CRITICAL: Create Series for target (REQUIRED by Bgolearn)
measured_response = pd.Series(strength, name='Tensile_Strength')

print(f"Dataset created: {len(data_matrix)} samples")
print(f"Features: {list(data_matrix.columns)}")
print(f"Target range: {measured_response.min():.1f} - {measured_response.max():.1f} MPa")

# Step 2: Create candidate materials for optimization
print("\nCreating candidate materials...")

# Generate candidate space
n_candidates = 100
candidates_data = {
    'Al_content': np.random.uniform(0.75, 0.98, n_candidates),
    'Cu_content': np.random.uniform(0.01, 0.20, n_candidates),
    'Mg_content': np.random.uniform(0.005, 0.10, n_candidates),
    'Temperature': np.random.uniform(350, 650, n_candidates),
    'Pressure': np.random.uniform(0.5, 6.0, n_candidates),
    'Time': np.random.uniform(0.5, 15, n_candidates)
}

# CRITICAL: Create DataFrame for candidates (REQUIRED by Bgolearn)
virtual_samples = pd.DataFrame(candidates_data)

print(f"Candidates created: {len(virtual_samples)} materials")

# Step 3: Initialize and fit Bgolearn with proper data types
print("\nFitting Bgolearn model...")

# Initialize optimizer
opt = BGOsampling.Bgolearn()

# CRITICAL: All inputs must be pandas DataFrame/Series
model = opt.fit(
    data_matrix=data_matrix,        # DataFrame with column names
    Measured_response=measured_response,  # Series with target values
    virtual_samples=virtual_samples,     # DataFrame with same columns
    Mission='Regression',           # Continuous target
    Classifier='GaussianProcess',   # Surrogate model
    opt_num=1,                     # Single recommendation
    min_search=False,              # False = maximization
    CV_test=5,                     # 5-fold cross-validation
    Dynamic_W=False,               # Static weights
    Normalize=True,                # Normalize features
    seed=42                        # Random seed
)

print("Optimization completed successfully!")

# Step 4: Get recommendations using Expected Improvement
print("\n=== Getting Recommendations ===")

# Get EI recommendation
ei_values, ei_points = model.EI()

print(f"Best recommendation:")
print(f"  Al: {ei_points[0][0]:.3f}")
print(f"  Cu: {ei_points[0][1]:.3f}")
print(f"  Mg: {ei_points[0][2]:.3f}")
print(f"  Temperature: {ei_points[0][3]:.1f}°C")
print(f"  Pressure: {ei_points[0][4]:.2f} GPa")
print(f"  Time: {ei_points[0][5]:.1f} hours")

# Get predicted strength
best_idx = np.argmax(ei_values)
predicted_strength = model.virtual_samples_mean[best_idx]
prediction_std = model.virtual_samples_std[best_idx]

print(f"  Predicted strength: {predicted_strength:.1f} ± {prediction_std:.1f} MPa")
print(f"  Expected Improvement: {ei_values[best_idx]:.4f}")

print("\n Materials discovery analysis completed!")
```

### Key Points for Success:

1. **Always use pandas DataFrames/Series**: Bgolearn requires pandas format, not numpy arrays
2. **Include column names**: DataFrames must have proper column names
3. **Match column names**: `virtual_samples` must have same columns as `data_matrix`
4. **Use Series for targets**: `Measured_response` should be a pandas Series
5. **Check data types**: Use `type(data_matrix)` to verify it's a DataFrame

### Best Practices for Materials Discovery

### Data Quality and Preprocessing

```python
def materials_data_preprocessing_tips():
    """Best practices for materials data preprocessing."""
    
    print("=== Materials Data Preprocessing Best Practices ===")
    
    # 1. Composition normalization
    print("1. Composition Normalization:")
    print("   - Ensure compositions sum to 1 (or known total)")
    print("   - Handle trace elements consistently")
    print("   - Use atomic fractions or weight fractions consistently")
    
    # 2. Feature scaling
    print("\n2. Feature Scaling:")
    print("   - Temperature: Often log-scale or normalized")
    print("   - Pressure: Consider log-scale for wide ranges")
    print("   - Time: Often log-scale for processing time")
    print("   - Compositions: Usually already normalized")
    
    # 3. Property transformations
    print("\n3. Property Transformations:")
    print("   - Strength: Usually linear scale")
    print("   - Ductility: May benefit from log transformation")
    print("   - Electrical properties: Often log-scale")
    print("   - Consider physical constraints (positive values)")
    
    # Example preprocessing
    X_processed = X_initial.copy()
    
    # Log-transform time (processing time often has log-normal distribution)
    X_processed['Log_Time'] = np.log(X_processed['Time'] + 0.1)
    
    # Normalize temperature to [0, 1] range
    temp_min, temp_max = 400, 800
    X_processed['Temp_Normalized'] = (X_processed['Temperature'] - temp_min) / (temp_max - temp_min)
    
    print(f"\nExample preprocessing applied:")
    print(f"Original time range: [{X_initial['Time'].min():.1f}, {X_initial['Time'].max():.1f}]")
    print(f"Log time range: [{X_processed['Log_Time'].min():.2f}, {X_processed['Log_Time'].max():.2f}]")
    print(f"Normalized temp range: [{X_processed['Temp_Normalized'].min():.2f}, {X_processed['Temp_Normalized'].max():.2f}]")

materials_data_preprocessing_tips()
```

### Experimental Design Strategies

```python
def experimental_design_strategies():
    """Strategies for effective experimental design in materials discovery."""
    
    print("=== Experimental Design Strategies ===")
    
    strategies = {
        "Space-filling initial design": [
            "Use Latin Hypercube Sampling for initial experiments",
            "Ensure good coverage of composition and processing space",
            "Include known good materials as benchmarks"
        ],
        
        "Acquisition function selection": [
            "EI: Good general choice, balances exploration/exploitation",
            "UCB: Better for noisy experiments, adjustable exploration",
            "Sequential optimization: Build experiments one at a time for maximum learning"
        ],
        
        "Constraint handling": [
            "Define feasible composition ranges early",
            "Consider processing equipment limitations",
            "Include cost constraints if relevant"
        ],
        
        "Multi-objective considerations": [
            "Identify property trade-offs early",
            "Use weighted objectives or Pareto optimization",
            "Consider manufacturing constraints"
        ],
        
        "Iterative refinement": [
            "Start with broad exploration, then focus",
            "Update models frequently with new data",
            "Monitor convergence and adjust strategy"
        ]
    }
    
    for strategy, tips in strategies.items():
        print(f"\n{strategy}:")
        for tip in tips:
            print(f"  • {tip}")

experimental_design_strategies()
```

### Model Validation for Materials

```python
def materials_model_validation():
    """Specific validation approaches for materials models."""
    
    print("=== Materials Model Validation ===")
    
    # Cross-validation with composition awareness
    from sklearn.model_selection import GroupKFold
    
    # Group by similar compositions for more realistic CV
    def create_composition_groups(X, n_groups=5):
        """Create groups based on composition similarity."""
        from sklearn.cluster import KMeans
        
        compositions = X[['Al_content', 'Cu_content', 'Mg_content']].values
        kmeans = KMeans(n_clusters=n_groups, random_state=42)
        groups = kmeans.fit_predict(compositions)
        return groups
    
    groups = create_composition_groups(X_initial)
    
    print(f"Created {len(np.unique(groups))} composition groups for CV")
    print("Group sizes:", [np.sum(groups == i) for i in np.unique(groups)])
    
    # Physics-based validation
    print("\nPhysics-based validation checks:")
    print("• Strength should generally increase with certain alloying elements")
    print("• Ductility often trades off with strength")
    print("• Temperature effects should follow known metallurgy")
    print("• Composition effects should be physically reasonable")
    
    # Extrapolation warnings
    print("\nExtrapolation detection:")
    print("• Monitor when predictions go outside training data range")
    print("• Check for unrealistic property combinations")
    print("• Validate against known materials databases")

materials_model_validation()
```

## Summary

This comprehensive materials discovery guide demonstrates:

1. **Problem Setup**: Realistic materials optimization with compositions and processing
2. **Single-Objective**: Optimizing individual properties like tensile strength
3. **Multi-Objective**: Balancing competing properties like strength vs ductility
4. **Constraints**: Handling composition and processing limitations
5. **Iterative Design**: Simulating real experimental workflows
6. **Best Practices**: Data preprocessing, validation, and experimental strategies

Bgolearn provides the tools needed for efficient materials discovery, helping researchers find optimal compositions and processing conditions with minimal experiments.
