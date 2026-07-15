# Getting Started with Bgolearn

```{note}
This guide will help you install Bgolearn and run your first optimization in just a few minutes.
```

## Installation

### Prerequisites

Bgolearn requires Python 3.7 or higher. We recommend using a virtual environment:

```bash
# Create virtual environment
python -m venv bgolearn_env

# Activate virtual environment
# On Windows:
bgolearn_env\Scripts\activate
# On macOS/Linux:
source bgolearn_env/bin/activate
```

### Install Bgolearn

Install the main package from PyPI:

```bash
pip install Bgolearn
```

For multi-objective optimization, also install MultiBgolearn:

```bash
pip install MultiBgolearn
```

Or install both together:

```bash
pip install Bgolearn MultiBgolearn
```

### Verify Installation

Test your installation:

```python
# Test single-objective Bgolearn
from Bgolearn import BGOsampling
print("Bgolearn imported successfully!")

# Test multi-objective MultiBgolearn
try:
    from MultiBgolearn import bgo
    print("MultiBgolearn imported successfully!")
except ImportError:
    print("MultiBgolearn not installed. Install with: pip install MultiBgolearn")
```

## Basic Concepts

### What is Bayesian Optimization?

Bayesian optimization is a powerful technique for optimizing expensive-to-evaluate functions. It's particularly useful when:
- Experiments are costly (time, money, resources)
- Function evaluations are noisy
- Gradients are unavailable
- You want to minimize the number of experiments

### Key Components

1. **Surrogate Model**: Approximates the unknown function (usually Gaussian Process etc.)
2. **Acquisition Function**: Decides where to sample next
3. **Optimization Loop**: Iteratively improves the surrogate model

## Your First Optimization

### Step 1: Generate Sample Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Generate synthetic materials data
def create_materials_dataset(n_samples=50, n_features=4, noise=0.1):
    """Create synthetic materials property data."""
    np.random.seed(42)
    
    # Generate base features
    X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                          noise=noise, random_state=42)
    
    # Create realistic feature names
    feature_names = ['Temperature', 'Pressure', 'Composition_A', 'Composition_B']
    
    # Normalize features to realistic ranges
    X[:, 0] = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min()) * 500 + 300  # Temperature: 300-800K
    X[:, 1] = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min()) * 10 + 1     # Pressure: 1-11 GPa
    X[:, 2] = np.abs(X[:, 2]) / np.abs(X[:, 2]).max() * 0.8 + 0.1  # Composition: 0.1-0.9
    X[:, 3] = 1 - X[:, 2]  # Ensure compositions sum to 1
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Strength'] = y  # Target property (e.g., material strength)
    
    return df

# Create training data
train_data = create_materials_dataset(n_samples=30)
print("Training data shape:", train_data.shape)
print("\nFirst 5 rows:")
print(train_data.head())
```

### Step 2: Prepare Data for Optimization

```python
from Bgolearn import BGOsampling

# Separate features and target
X_train = train_data.drop('Strength', axis=1)
y_train = train_data['Strength']

# Create virtual candidates for optimization
def create_candidate_materials(n_candidates=200):
    """Create candidate materials for optimization."""
    np.random.seed(123)
    
    # Generate candidate space
    candidates = []
    for _ in range(n_candidates):
        temp = np.random.uniform(300, 800)  # Temperature
        pressure = np.random.uniform(1, 11)  # Pressure
        comp_a = np.random.uniform(0.1, 0.9)  # Composition A
        comp_b = 1 - comp_a  # Composition B
        
        candidates.append([temp, pressure, comp_a, comp_b])
    
    return pd.DataFrame(candidates, columns=X_train.columns)

X_candidates = create_candidate_materials()
print(f"Created {len(X_candidates)} candidate materials")
```

### Step 3: Initialize and Fit Bgolearn

```python
# Initialize Bgolearn optimizer
optimizer = BGOsampling.Bgolearn()

# Fit the model
print("Fitting Bgolearn model...")
model = optimizer.fit(
    data_matrix=X_train,
    Measured_response=y_train,
    virtual_samples=X_candidates,
    Mission='Regression',
    min_search=False,  # We want to maximize strength
    CV_test=5,  # 5-fold cross-validation
    Normalize=True
)

print("Model fitted successfully!")
```

### Step 4: Single-Point Optimization

```python
# Expected Improvement
print("\n=== Expected Improvement ===")
ei_values, next_point_ei = model.EI()
print(f"Next experiment (EI): {next_point_ei}")

# Upper Confidence Bound
print("\n=== Upper Confidence Bound ===")
ucb_values, next_point_ucb = model.UCB(alpha=2.0)
print(f"Next experiment (UCB): {next_point_ucb}")

# Probability of Improvement
print("\n=== Probability of Improvement ===")
poi_values, next_point_poi = model.PoI(tao=0.01)
print(f"Next experiment (PoI): {next_point_poi}")
```


### Step 5: Basic Visualization

```python
import matplotlib.pyplot as plt

# Get EI values for visualization
ei_values, recommended_points = model.EI()

# Plot Expected Improvement values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(ei_values)
plt.title("Expected Improvement Values")
plt.xlabel("Candidate Index")
plt.ylabel("EI Value")
plt.grid(True)

# Plot predicted vs actual (if you have true function)
plt.subplot(1, 2, 2)
plt.scatter(range(len(model.virtual_samples_mean)), model.virtual_samples_mean, alpha=0.6)
plt.title("Predicted Values for All Candidates")
plt.xlabel("Candidate Index")
plt.ylabel("Predicted Value")
plt.grid(True)

plt.tight_layout()
plt.show()

```

## Understanding the Results

### Interpreting Acquisition Functions

1. **Expected Improvement (EI)**:
   - Balances exploration and exploitation
   - Higher values indicate more promising regions
   - Good general-purpose choice

2. **Upper Confidence Bound (UCB)**:
   - Controlled by parameter `alpha`
   - Higher `alpha` = more exploration
   - Good for noisy functions

3. **Probability of Improvement (PoI)**:
   - Simple and intuitive
   - Controlled by parameter `tao`
   - Can be overly exploitative

### Model Validation

```python
# Check cross-validation results
print("\n=== Model Validation ===")
print("Cross-validation results saved in ./Bgolearn/ directory")

# List generated files
import os
bgo_files = [f for f in os.listdir('./Bgolearn') if f.endswith('.csv')]
print("Generated files:")
for file in bgo_files:
    print(f"  - {file}")
```

## Next Steps

Now that you've completed your first optimization, explore these advanced topics:

1. **[Acquisition Functions](acquisition_functions.md)** - Deep dive into different acquisition strategies
2. **[Batch Optimization](batch_optimization.md)** - Parallel experiment design
3. **[Visualization](visualization.md)** - Advanced plotting and dashboards
4. **[Materials Discovery](materials_discovery.md)** - Specialized workflows

## Tips for Success

### Data Preparation
- Ensure features are on similar scales (use `Normalize=True`)
- Remove highly correlated features
- Handle missing values appropriately

### Model Selection
- Start with default Gaussian Process
- Use cross-validation to assess model quality
- Consider noise levels in your experiments

### Acquisition Function Choice
- EI: Good general choice
- UCB: Better for noisy functions
- Batch methods: When you can run parallel experiments

### Iteration Strategy
- Start with space-filling design
- Use acquisition functions for refinement
- Monitor convergence carefully

## Common Issues

### Memory Issues
```python
# For large candidate sets, use batching
if len(X_candidates) > 100000:
    print("Large candidate set detected. Consider using smaller batches.")
```

### Convergence Problems
```python
# Check for proper normalization
print("Feature ranges:")
print(X_train.describe())

# Ensure sufficient training data
if len(X_train) < 10:
    print("Warning: Very small training set. Consider collecting more data.")
```

### Numerical Stability
```python
# Check for extreme values
print("Target variable statistics:")
print(y_train.describe())

# Look for outliers
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
outliers = y_train[(y_train < Q1 - 1.5*IQR) | (y_train > Q3 + 1.5*IQR)]
print(f"Potential outliers: {len(outliers)}")
```
