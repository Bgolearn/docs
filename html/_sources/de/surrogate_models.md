# Surrogate Models in Bgolearn

```{note}
Diese Seite ist die deutsche Version des Bgolearn-Handbuchs.
```

```{warning}
**Important**: All data must be pandas DataFrames/Series, not numpy arrays!

- `data_matrix` → `pd.DataFrame` with column names
- `Measured_response` → `pd.Series`
- `virtual_samples` → `pd.DataFrame` with same column names

Using numpy arrays will cause: `AttributeError: 'numpy.ndarray' object has no attribute 'columns'`
```

```{note}
This page explains the different surrogate models available in Bgolearn and how to choose the right one for your optimization problem.
```

## Overview

Surrogate models (also called metamodels) are the core of Bayesian optimization. They approximate the expensive-to-evaluate objective function and provide uncertainty estimates that guide the optimization process.

Bgolearn supports several surrogate models, each with different strengths and use cases:

1. **Gaussian Process (GP)** - Default and most common
2. **Random Forest (RF)** - Good for discrete/categorical features
3. **Support Vector Regression (SVR)** - Robust to noise
4. **Multi-Layer Perceptron (MLP)** - Neural network approach
5. **AdaBoost** - Ensemble method

## Gaussian Process (GaussianProcess)

### Theory

Gaussian Processes are the gold standard for Bayesian optimization. They provide both predictions and uncertainty estimates in a principled way.

```python
from Bgolearn import BGOsampling

# Use Gaussian Process (default)
opt = BGOsampling.Bgolearn()
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Classifier='GaussianProcess'  # Explicit specification
)
```

### Advantages

```{admonition} GP Strengths
:class: tip
- **Uncertainty Quantification**: Provides prediction uncertainty
- **Theoretical Foundation**: Well-established Bayesian framework
- **Smooth Interpolation**: Works well for continuous functions
- **Few Hyperparameters**: Relatively easy to tune
- **Global Optimization**: Good for finding global optima
```

### Limitations

```{admonition} GP Limitations
:class: warning
- **Computational Cost**: O(n³) scaling with training data
- **Smoothness Assumption**: Assumes smooth underlying function
- **High-Dimensional Challenges**: Struggles with many features (>20)
- **Categorical Features**: Not naturally suited for discrete variables
```

### Best Use Cases

- **Continuous optimization problems**
- **Smooth objective functions**
- **Small to medium datasets** (<1000 samples)
- **When uncertainty is important**
- **Materials property optimization**

### Example: Alloy Optimization with GP

```python
import numpy as np
import pandas as pd
import copy
from Bgolearn import BGOsampling

# Alloy composition optimization - use pandas DataFrame
data_matrix = pd.DataFrame([
    [2.0, 1.2, 0.5],  # Cu, Mg, Si
    [3.5, 0.8, 0.7],
    [1.8, 1.5, 0.3],
    [4.2, 0.9, 0.8]
], columns=['Cu', 'Mg', 'Si'])

strength_values = pd.Series([250, 280, 240, 290])  # MPa
measured_response = copy.deepcopy(strength_values)


virtual_samples = pd.DataFrame([
    [2.5, 1.0, 0.6],
    [3.0, 1.3, 0.4],
    [3.8, 0.9, 0.8]
], columns=['Cu', 'Mg', 'Si'])

# GP optimization
opt = BGOsampling.Bgolearn()
model = opt.fit(
    data_matrix=data_matrix,  # DataFrame
    Measured_response=strength_values,  # Series
    virtual_samples=virtual_samples,  # DataFrame
    Classifier='GaussianProcess',
    CV_test=2,  # 2-fold cross-validation
    Normalize=True
)

print("GP optimization completed!")
```

## Random Forest (RandomForest)

### Theory

Random Forest builds multiple decision trees and averages their predictions. It's particularly good for handling discrete features and non-linear relationships.

```python
# Use Random Forest
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Classifier='RandomForest'
)
```

### Advantages

```{admonition} RF Strengths
:class: tip
- **Handles Discrete Features**: Natural for categorical variables
- **Non-linear Relationships**: Captures complex patterns
- **Robust to Outliers**: Less sensitive to noisy data
- **Fast Training**: Efficient for large datasets
- **Feature Importance**: Provides feature ranking
- **No Assumptions**: Doesn't assume function smoothness
```

### Limitations

```{admonition} RF Limitations
:class: warning
- **Limited Uncertainty**: Poor uncertainty quantification
- **Overfitting Risk**: Can overfit with small datasets
- **Discontinuous**: Creates step-wise predictions
- **Hyperparameter Tuning**: Many parameters to optimize
```

### Best Use Cases

- **Discrete/categorical features**
- **Large datasets** (>1000 samples)
- **Non-smooth functions**
- **When robustness is important**
- **Mixed variable types**

### Example: Processing Parameter Optimization

```python
# Processing parameters with discrete levels - use pandas DataFrame
processing_data = pd.DataFrame([
    [450, 2, 1],    # Temperature, Time, Atmosphere (1=N2, 2=Ar, 3=Air)
    [500, 4, 2],
    [550, 6, 3],
    [480, 3, 1]
], columns=['Temperature', 'Time', 'Atmosphere'])

hardness_values = pd.Series([180, 220, 250, 200])

virtual_processing = pd.DataFrame([
    [475, 3.5, 1],
    [525, 4.5, 2],
    [490, 2.5, 3]
], columns=['Temperature', 'Time', 'Atmosphere'])

# Random Forest for mixed variables
model = opt.fit(
    data_matrix=processing_data,  # DataFrame
    Measured_response=hardness_values,  # Series
    virtual_samples=virtual_processing,  # DataFrame
    Classifier='RandomForest',
    CV_test='LOOCV'  # Leave-one-out cross-validation
)

print("Random Forest optimization completed!")
```

## Support Vector Regression (SVR)

### Theory

SVR uses support vector machines for regression, finding a function that deviates from targets by at most ε while being as flat as possible.

```python
# Use SVR
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Classifier='SVR'
)
```

### Advantages

```{admonition} SVR Strengths
:class: tip
- **Robust to Noise**: Handles noisy data well
- **High-Dimensional**: Works with many features
- **Kernel Flexibility**: Different kernel functions available
- **Sparse Solution**: Uses only support vectors
- **Regularization**: Built-in overfitting protection
```

### Limitations

```{admonition} SVR Limitations
:class: warning
- **Parameter Sensitivity**: Requires careful tuning
- **No Uncertainty**: Doesn't provide prediction uncertainty
- **Kernel Selection**: Choosing the right kernel is crucial
- **Computational Cost**: Can be slow for large datasets
```

### Best Use Cases

- **Noisy data**
- **High-dimensional problems**
- **When robustness is critical**
- **Non-linear relationships**

### Example: High-Dimensional Optimization

```python
# High-dimensional alloy with many elements - use pandas DataFrame
element_names = ['Al', 'Cu', 'Mg', 'Si', 'Fe', 'Mn', 'Cr', 'Zn']
high_dim_data = pd.DataFrame(
    np.random.random((20, 8)),  # 8 alloying elements
    columns=element_names
)
high_dim_response = pd.Series(np.random.random(20) * 100 + 200)

high_dim_virtual = pd.DataFrame(
    np.random.random((50, 8)),
    columns=element_names
)

# SVR for high-dimensional problem
model = opt.fit(
    data_matrix=high_dim_data,  # DataFrame
    Measured_response=high_dim_response,  # Series
    virtual_samples=high_dim_virtual,  # DataFrame
    Classifier='SVR',
    Normalize=True
)

print("SVR optimization completed!")
```

## Multi-Layer Perceptron (MLP)

### Theory

MLP is a neural network with multiple hidden layers that can approximate complex non-linear functions.

```python
# Use MLP
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Classifier='MLP'
)
```

### Advantages

```{admonition} MLP Strengths
:class: tip
- **Universal Approximator**: Can approximate any continuous function
- **Non-linear Modeling**: Excellent for complex relationships
- **Scalable**: Can handle large datasets
- **Flexible Architecture**: Customizable network structure
```

### Limitations

```{admonition} MLP Limitations
:class: warning
- **Requires Large Data**: Needs substantial training data
- **Hyperparameter Tuning**: Many parameters to optimize
- **Overfitting Risk**: Can easily overfit small datasets
- **No Uncertainty**: Doesn't provide uncertainty estimates
- **Training Instability**: Can be sensitive to initialization
```

### Best Use Cases

- **Large datasets** (>500 samples)
- **Complex non-linear relationships**
- **When sufficient data is available**
- **Pattern recognition tasks**

## AdaBoost

### Theory

AdaBoost (Adaptive Boosting) combines multiple weak learners to create a strong predictor.

```python
# Use AdaBoost
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Classifier='AdaBoost'
)
```

### Advantages

```{admonition} AdaBoost Strengths
:class: tip
- **Ensemble Method**: Combines multiple models
- **Adaptive**: Focuses on difficult examples
- **Reduces Bias**: Often improves prediction accuracy
- **Versatile**: Works with different base learners
```

### Limitations

```{admonition} AdaBoost Limitations
:class: warning
- **Sensitive to Noise**: Can overfit noisy data
- **Computational Cost**: Slower than single models
- **Parameter Tuning**: Requires careful tuning
- **Limited Uncertainty**: Poor uncertainty quantification
```

## Model Selection Guidelines

### Decision Tree

```
Start Here
    ↓
Do you have uncertainty requirements?
    ↓ Yes                    ↓ No
Gaussian Process         What type of features?
    ↓                        ↓
Is data smooth?         Continuous → SVR/MLP
    ↓ Yes    ↓ No           ↓
    GP    Random Forest   Discrete → Random Forest
                             ↓
                        Large dataset?
                             ↓ Yes    ↓ No
                            MLP    Random Forest
```

### Detailed Selection Criteria

```{list-table} Model Selection Guide
:header-rows: 1
:name: model-selection

* - Criterion
  - Gaussian Process
  - Random Forest
  - SVR
  - MLP
  - AdaBoost
* - **Data Size**
  - <1000
  - >500
  - >500
  - >500
  - >500
* - **Feature Types**
  - Continuous
  - Mixed
  - Continuous
  - Continuous
  - Mixed
* - **Uncertainty**
  - Excellent
  - Poor
  - None
  - None
  - Poor
* - **Noise Tolerance**
  - Medium
  - High
  - High
  - Medium
  - Low
* - **Training Speed**
  - Slow
  - Fast
  - Medium
  - Slow
  - Medium
* - **Prediction Speed**
  - Fast
  - Fast
  - Fast
  - Fast
  - Medium
```

## Practical Examples

### Model Comparison

```python
# Create test data for model comparison
data_matrix = pd.DataFrame([
    [2.0, 1.2, 0.5],
    [3.5, 0.8, 0.7],
    [1.8, 1.5, 0.3],
    [4.2, 0.9, 0.8],
    [2.8, 1.1, 0.6]
], columns=['Cu', 'Mg', 'Si'])

measured_response = pd.Series([250, 280, 240, 290, 265])

virtual_samples = pd.DataFrame([
    [2.5, 1.0, 0.6],
    [3.0, 1.3, 0.4],
    [3.8, 0.9, 0.8]
], columns=['Cu', 'Mg', 'Si'])

# Compare all models on the same problem
models = ['GaussianProcess', 'RandomForest', 'SVR', 'MLP', 'AdaBoost']
results = {}

for model_name in models:
    print(f"Testing {model_name}...")

    try:
        model = opt.fit(
            data_matrix=data_matrix,  # DataFrame
            Measured_response=measured_response,  # Series
            virtual_samples=virtual_samples,  # DataFrame
            Classifier=model_name,
            CV_test=5,  # 5-fold cross-validation
            Normalize=True
        )
        
        # Get results using EI
        ei_values, recommended_points = model.EI()
        recommended_point = recommended_points[0]

        # Get model performance metrics (if available)
        try:
            # Access cross-validation score if available
            cv_score = getattr(model, 'cv_score_', 0.0)
        except:
            cv_score = 0.0

        results[model_name] = {
            'recommended_point': recommended_point,
            'ei_max': np.max(ei_values),
            'cv_score': cv_score,
            'success': True
        }

        print(f"  Success: EI max = {np.max(ei_values):.3f}")
        
    except Exception as e:
        print(f"  Failed: {str(e)}")
        results[model_name] = {'success': False, 'error': str(e)}

# Display comparison
print("\nModel Performance Comparison:")
print("-" * 50)
for model, result in results.items():
    if result['success']:
        print(f"{model:15s}: Success - EI max = {result['ei_max']:.3f}")
    else:
        print(f"{model:15s}: Failed - {result.get('error', 'Unknown error')}")
```

### Hyperparameter Tuning

```python
# Hyperparameter tuning in Bgolearn
import numpy as np
import pandas as pd
from Bgolearn import BGOsampling

# Create test data
data_matrix = pd.DataFrame([
    [2.0, 1.2, 0.5, 450],  # Cu, Mg, Si, Temperature
    [3.5, 0.8, 0.7, 500],
    [1.8, 1.5, 0.3, 480],
    [4.2, 0.9, 0.8, 520],
    [2.8, 1.1, 0.6, 490],
    [3.2, 1.3, 0.4, 510],
    [2.5, 0.9, 0.7, 470],
    [3.8, 1.0, 0.5, 530]
], columns=['Cu', 'Mg', 'Si', 'Temperature'])

strength_values = pd.Series([250, 280, 240, 290, 265, 275, 245, 295])

virtual_samples = pd.DataFrame([
    [2.5, 1.0, 0.6, 485],
    [3.0, 1.3, 0.4, 505],
    [3.8, 0.9, 0.8, 515]
], columns=['Cu', 'Mg', 'Si', 'Temperature'])

# Hyperparameter tuning: compare different cross-validation settings
cv_settings = [3, 5, 10, 'LOOCV']
normalization_settings = [True, False]

best_score = -np.inf
best_config = None
results = {}

opt = BGOsampling.Bgolearn()

print("Starting hyperparameter tuning...")
print("=" * 50)

for cv_test in cv_settings:
    for normalize in normalization_settings:
        config_name = f"CV_{cv_test}_Norm_{normalize}"
        print(f"Testing configuration: {config_name}")

        try:
            model = opt.fit(
                data_matrix=data_matrix,
                Measured_response=strength_values,
                virtual_samples=virtual_samples,
                Classifier='GaussianProcess',
                CV_test=cv_test,
                Normalize=normalize,
                seed=42  # Ensure reproducibility
            )

            # Get EI values as performance metric
            ei_values, recommended_points = model.EI()
            max_ei = np.max(ei_values)

            # Calculate prediction quality
            predicted_mean = model.virtual_samples_mean
            predicted_std = model.virtual_samples_std

            # Combined score: EI max + inverse of prediction uncertainty
            uncertainty_score = 1.0 / (np.mean(predicted_std) + 1e-6)
            combined_score = max_ei + 0.1 * uncertainty_score

            results[config_name] = {
                'max_ei': max_ei,
                'mean_uncertainty': np.mean(predicted_std),
                'combined_score': combined_score,
                'recommended_point': recommended_points[0],
                'success': True
            }

            if combined_score > best_score:
                best_score = combined_score
                best_config = config_name

            print(f"  Success - EI max: {max_ei:.3f}, Combined score: {combined_score:.3f}")

        except Exception as e:
            print(f"  Failed: {str(e)}")
            results[config_name] = {'success': False, 'error': str(e)}

# Display tuning results
print(f"\n Hyperparameter tuning results:")
print("=" * 50)
print(f"{'Configuration':<20} {'EI Max':<10} {'Mean Uncertainty':<15} {'Combined Score':<15}")
print("-" * 60)

for config, result in results.items():
    if result['success']:
        print(f"{config:<20} {result['max_ei']:<10.3f} {result['mean_uncertainty']:<15.3f} {result['combined_score']:<15.3f}")
    else:
        print(f"{config:<20} {'Failed':<10} {'-':<15} {'-':<15}")

print(f"\n Best configuration: {best_config}")
print(f" Best score: {best_score:.3f}")

if best_config and results[best_config]['success']:
    best_point = results[best_config]['recommended_point']
    print(f" Next experiment point recommended by best configuration:")
    print(f"   Cu: {best_point[0]:.2f}%, Mg: {best_point[1]:.2f}%, Si: {best_point[2]:.2f}%, T: {best_point[3]:.0f}K")
```

## Advanced Topics

### Ensemble Methods

Implement ensemble methods in Bgolearn by combining multiple surrogate models for improved performance:

```python
import numpy as np
import pandas as pd
from Bgolearn import BGOsampling

# Create test data
data_matrix = pd.DataFrame([
    [2.0, 1.2, 0.5],
    [3.5, 0.8, 0.7],
    [1.8, 1.5, 0.3],
    [4.2, 0.9, 0.8],
    [2.8, 1.1, 0.6],
    [3.2, 1.3, 0.4]
], columns=['Cu', 'Mg', 'Si'])

strength_values = pd.Series([250, 280, 240, 290, 265, 275])

virtual_samples = pd.DataFrame([
    [2.5, 1.0, 0.6],
    [3.0, 1.3, 0.4],
    [3.8, 0.9, 0.8],
    [2.2, 1.4, 0.5],
    [3.6, 0.7, 0.6]
], columns=['Cu', 'Mg', 'Si'])

# Define ensemble method class
class BgolearnEnsemble:
    """Bgolearn ensemble method implementation"""

    def __init__(self, model_types=['GaussianProcess', 'RandomForest', 'SVR']):
        self.model_types = model_types
        self.models = {}
        self.weights = None

    def fit(self, data_matrix, measured_response, virtual_samples, **kwargs):
        """Train all models"""
        print(" Training ensemble models...")

        opt = BGOsampling.Bgolearn()

        for model_type in self.model_types:
            print(f"  Training {model_type}...")
            try:
                model = opt.fit(
                    data_matrix=data_matrix,
                    Measured_response=measured_response,
                    virtual_samples=virtual_samples,
                    Classifier=model_type,
                    CV_test=5,
                    Normalize=True,
                    **kwargs
                )
                self.models[model_type] = model
                print(f"  {model_type} training successful")
            except Exception as e:
                print(f"  {model_type} training failed: {e}")

        # Calculate model weights (based on EI performance)
        self._calculate_weights()

    def _calculate_weights(self):
        """Calculate model weights based on EI performance"""
        if not self.models:
            return

        ei_scores = {}
        for name, model in self.models.items():
            try:
                ei_values, _ = model.EI()
                ei_scores[name] = np.max(ei_values)
            except:
                ei_scores[name] = 0.0

        # Normalize weights
        total_score = sum(ei_scores.values())
        if total_score > 0:
            self.weights = {name: score/total_score for name, score in ei_scores.items()}
        else:
            # Equal weights
            self.weights = {name: 1.0/len(self.models) for name in self.models.keys()}

        print(f"Model weights: {self.weights}")

    def ensemble_EI(self):
        """Ensemble expected improvement"""
        if not self.models:
            raise ValueError("No trained models available")

        ensemble_ei = None
        ensemble_points = None

        for name, model in self.models.items():
            try:
                ei_values, points = model.EI()
                weight = self.weights.get(name, 0.0)

                if ensemble_ei is None:
                    ensemble_ei = weight * ei_values
                    ensemble_points = points
                else:
                    ensemble_ei += weight * ei_values

            except Exception as e:
                print(f" {name} EI calculation failed: {e}")

        # Find the point corresponding to maximum EI
        if ensemble_ei is not None:
            max_idx = np.argmax(ensemble_ei)
            best_point = ensemble_points[max_idx:max_idx+1]  # Maintain dimensions
            return ensemble_ei, best_point
        else:
            raise ValueError("All model EI calculations failed")

    def ensemble_predictions(self):
        """Ensemble predictions"""
        predictions = {}
        uncertainties = {}

        for name, model in self.models.items():
            try:
                pred_mean = model.virtual_samples_mean
                pred_std = model.virtual_samples_std

                predictions[name] = pred_mean
                uncertainties[name] = pred_std

            except Exception as e:
                print(f" {name} prediction failed: {e}")

        if not predictions:
            raise ValueError("All model predictions failed")

        # Weighted average predictions
        ensemble_mean = np.zeros_like(list(predictions.values())[0])
        ensemble_std = np.zeros_like(list(uncertainties.values())[0])

        for name, pred in predictions.items():
            weight = self.weights.get(name, 0.0)
            ensemble_mean += weight * pred
            ensemble_std += weight * uncertainties[name]

        return ensemble_mean, ensemble_std

    def get_model_comparison(self):
        """Get model comparison results"""
        comparison = {}

        for name, model in self.models.items():
            try:
                ei_values, points = model.EI()
                pred_mean = model.virtual_samples_mean
                pred_std = model.virtual_samples_std

                comparison[name] = {
                    'max_ei': np.max(ei_values),
                    'mean_prediction': np.mean(pred_mean),
                    'mean_uncertainty': np.mean(pred_std),
                    'weight': self.weights.get(name, 0.0),
                    'recommended_point': points[0]
                }
            except Exception as e:
                comparison[name] = {'error': str(e)}

        return comparison

# Using ensemble methods
print("Starting ensemble method demonstration")
print("=" * 50)

# Create and train ensemble model
ensemble = BgolearnEnsemble(
    model_types=['GaussianProcess', 'RandomForest', 'SVR']
)

ensemble.fit(
    data_matrix=data_matrix,
    measured_response=strength_values,
    virtual_samples=virtual_samples,
    seed=42
)

# Get ensemble predictions
print("\n Ensemble prediction results:")
try:
    ensemble_mean, ensemble_std = ensemble.ensemble_predictions()
    print(f"Ensemble prediction mean: {ensemble_mean[:3]}")  # Show first 3
    print(f"Ensemble prediction std: {ensemble_std[:3]}")
except Exception as e:
    print(f"Ensemble prediction failed: {e}")

# Get ensemble EI
print("\n Ensemble Expected Improvement:")
try:
    ensemble_ei, ensemble_point = ensemble.ensemble_EI()
    max_ei_idx = np.argmax(ensemble_ei)
    print(f"Maximum ensemble EI value: {ensemble_ei[max_ei_idx]:.3f}")
    print(f"Recommended next experiment point: Cu={ensemble_point[0][0]:.2f}, Mg={ensemble_point[0][1]:.2f}, Si={ensemble_point[0][2]:.2f}")
except Exception as e:
    print(f"Ensemble EI calculation failed: {e}")

# Model comparison
print("\n Model performance comparison:")
comparison = ensemble.get_model_comparison()
print(f"{'Model':<15} {'Max EI':<10} {'Weight':<8} {'Mean Uncertainty':<15}")
print("-" * 55)

for name, metrics in comparison.items():
    if 'error' not in metrics:
        print(f"{name:<15} {metrics['max_ei']:<10.3f} {metrics['weight']:<8.3f} {metrics['mean_uncertainty']:<15.3f}")
    else:
        print(f"{name:<15} {'Failed':<10} {'-':<8} {'-':<15}")

print("\n Ensemble method advantages:")
print("  - Combines strengths of multiple models")
print("  - Improves prediction stability")
print("  - Reduces single model bias")
print("  - Automatic weight allocation")
```

### Transfer Learning

Use pre-trained models for related problems:

```python
# Conceptual transfer learning
def transfer_learning(source_model, target_data, target_response):
    """Transfer knowledge from source to target problem."""
    # Use source model as initialization
    # Fine-tune on target data
    pass
```

### Online Learning

Update models with new data:

```python
# Conceptual online learning
def online_update(model, new_x, new_y):
    """Update model with new observation."""
    # Add new data point
    # Retrain or update model incrementally
    pass
```

## Troubleshooting

### Common Issues

1. **Poor CV Scores**
   - Try different models
   - Check data quality
   - Increase training data
   - Adjust normalization

2. **Slow Training**
   - Use Random Forest for large data
   - Reduce virtual space size
   - Consider SVR for high dimensions

3. **Overfitting**
   - Use cross-validation
   - Reduce model complexity
   - Add more training data

4. **Poor Uncertainty**
   - Use Gaussian Process
   - Increase bootstrap iterations
   - Check model assumptions

### Performance Optimization

```python
# Tips for better performance
optimization_tips = {
    "Data preprocessing": "Normalize features, remove outliers",
    "Model selection": "Start with GP, try RF for discrete features",
    "Hyperparameters": "Use cross-validation for tuning",
    "Computational": "Reduce virtual space size for speed",
    "Validation": "Use CV_test=10 or 'LOOCV' for validation"
}
```

## Next Steps

- **Learn acquisition functions**: {doc}`acquisition_functions`
- **Try optimization strategies**: {doc}`optimization_strategies`
- **Practice with examples**: {doc}`examples/single_objective`
- **Explore multi-objective**: {doc}`multibgolearn`

```{seealso}
For more on surrogate modeling:
- Rasmussen, C.E. "Gaussian Processes for Machine Learning"
- Forrester, A. "Engineering Design via Surrogate Modelling"
- Queipo, N.V. "Surrogate-based Analysis and Optimization"
```
