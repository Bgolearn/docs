# Bgolearn Examples

This directory contains comprehensive examples demonstrating various aspects of Bgolearn usage.

## 📁 Example Files

### Basic Examples
- **[basic_examples.py](basic_examples.py)** - Fundamental usage patterns
  - Simple 1D optimization
  - 2D function optimization
  - Batch optimization
  - Acquisition function comparison
  - Iterative optimization loop

### Materials Discovery Examples
- **[materials_examples.py](materials_examples.py)** - Materials science applications
  - Alloy composition optimization
  - Multi-objective materials design
  - Constraint handling
  - Processing parameter optimization

### Advanced Examples
- **[advanced_examples.py](advanced_examples.py)** - Complex scenarios
  - Multi-fidelity optimization
  - Custom acquisition functions
  - Large-scale optimization
  - Integration with other tools

### Benchmark Examples
- **[benchmark_examples.py](benchmark_examples.py)** - Performance comparisons
  - Standard test functions
  - Algorithm comparisons
  - Scalability analysis

## 🚀 Quick Start

### Running Basic Examples

```bash
# Run all basic examples
python basic_examples.py

# Or run individual examples in Python
python -c "
from basic_examples import example_1_simple_optimization
model, X, y, candidates = example_1_simple_optimization()
"
```

### Running Materials Examples

```bash
# Run materials discovery examples
python materials_examples.py

# Focus on specific materials problems
python -c "
from materials_examples import alloy_optimization_example
results = alloy_optimization_example()
"
```

## 📊 Example Outputs

Each example generates:
- **Visualization plots** (PNG files)
- **Console output** with optimization results
- **Model objects** for further analysis
- **Data files** (CSV) with results

## 🎯 Learning Path

### Beginner (Start Here)
1. `basic_examples.py` - Example 1: Simple 1D optimization
2. `basic_examples.py` - Example 2: 2D optimization
3. `materials_examples.py` - Basic alloy optimization

### Intermediate
1. `basic_examples.py` - Example 3: Batch optimization
2. `materials_examples.py` - Multi-objective optimization
3. `advanced_examples.py` - Custom acquisition functions

### Advanced
1. `advanced_examples.py` - Multi-fidelity optimization
2. `benchmark_examples.py` - Performance analysis
3. Custom implementations based on examples

## 🔧 Requirements

### Basic Requirements
```bash
pip install bgolearn numpy pandas matplotlib scikit-learn
```

### Advanced Features
```bash
pip install plotly seaborn  # For interactive visualizations
pip install pymatgen matminer  # For materials science features
```

## 📝 Example Structure

Each example follows this structure:

```python
def example_name():
    """
    Brief description of what this example demonstrates.
    """
    print("Example: Description")
    
    # 1. Data preparation
    X_train, y_train = create_data()
    X_candidates = create_candidates()
    
    # 2. Model fitting
    optimizer = BGOsampling.Bgolearn()
    model = optimizer.fit(X_train, y_train, X_candidates)
    
    # 3. Optimization
    values, next_point = model.EI()
    
    # 4. Visualization
    create_plots()
    
    # 5. Return results
    return model, results
```

## 🎨 Customization

### Modifying Examples

You can easily modify examples for your specific use case:

```python
# Change the objective function
def my_objective_function(x):
    return your_function(x)

# Modify data generation
X_train, y_train = create_your_data()

# Adjust optimization parameters
model = optimizer.fit(
    X_train, y_train, X_candidates,
    min_search=True,  # Your optimization direction
    CV_test=5,        # Your CV folds
    Normalize=True    # Your normalization preference
)

# Use different acquisition functions
ei_values, ei_point = model.EI()
ucb_values, ucb_point = model.UCB(alpha=2.0)
batch_indices, batch_points = model.batch_EI(q=5)
```

### Creating New Examples

Template for new examples:

```python
def my_new_example():
    """
    Description of your new example.
    """
    from Bgolearn import BGOsampling
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Your implementation here
    
    return results

if __name__ == "__main__":
    results = my_new_example()
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure Bgolearn is installed
   pip install bgolearn
   
   # Or install from source
   pip install -e .
   ```

2. **Visualization Issues**
   ```bash
   # Install visualization dependencies
   pip install matplotlib seaborn plotly
   ```

3. **Memory Issues with Large Examples**
   ```python
   # Reduce candidate set size
   X_candidates = X_candidates[:1000]  # Use fewer candidates
   
   # Use smaller batch sizes
   batch_result = model.batch_EI(q=3, mc_samples=500)
   ```

4. **Convergence Issues**
   ```python
   # Check data normalization
   model = optimizer.fit(..., Normalize=True)
   
   # Verify data quality
   print(X_train.describe())
   print(y_train.describe())
   ```

## 📚 Additional Resources

- **[API Reference](../api_reference.md)** - Complete function documentation
- **[Acquisition Functions Guide](../acquisition_functions.md)** - Detailed acquisition function explanations
- **[Visualization Guide](../visualization.md)** - Comprehensive plotting examples
- **[Materials Discovery Guide](../materials_discovery.md)** - Specialized materials workflows

## 🤝 Contributing Examples

We welcome new examples! To contribute:

1. Create your example following the template
2. Add comprehensive documentation
3. Include visualization and error handling
4. Test with different data sizes
5. Submit a pull request

### Example Contribution Template

```python
def contributed_example():
    """
    Brief description of the example.
    
    This example demonstrates:
    - Feature 1
    - Feature 2
    - Use case 3
    
    Returns
    -------
    results : dict
        Dictionary containing example results
    """
    # Implementation
    pass
```

## 📞 Support

If you have questions about the examples:

- Check the [documentation](../index.md)
- Open an issue on GitHub
- Contact: binjacobcao@gmail.com

Happy optimizing! 🚀
