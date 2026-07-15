# Bgolearn Documentation

Welcome to the comprehensive documentation for **Bgolearn** - a powerful Bayesian Global Optimization package designed specifically for materials discovery and scientific research.

## 🎯 What is Bgolearn?

Bgolearn is a state-of-the-art Python package that implements Bayesian optimization techniques to accelerate scientific discovery, particularly in materials science. It provides:

- **Multiple acquisition functions** for intelligent experiment selection
- **Batch optimization** for parallel experimental design
- **Advanced visualization tools** for understanding optimization progress
- **Materials-specific workflows** for composition and processing optimization
- **Robust error handling** and data validation

## 📚 Documentation Structure

### 🚀 Getting Started
- **[Installation & Quick Start](getting_started.md)** - Get up and running in minutes
- **[Basic Concepts](getting_started.md#basic-concepts)** - Understanding Bayesian optimization
- **[Your First Optimization](getting_started.md#your-first-optimization)** - Step-by-step tutorial

### 📖 Core Documentation
- **[API Reference](api_reference.md)** - Complete function and class documentation
- **[Acquisition Functions](acquisition_functions.md)** - Detailed guide to all acquisition strategies
- **[Batch Optimization](batch_optimization.md)** - Parallel experiment design
- **[Visualization](visualization.md)** - Comprehensive plotting and dashboard creation

### 🔬 Specialized Guides
- **[Materials Discovery](materials_discovery.md)** - Materials science workflows and examples
- **[Custom Models](custom_models.md)** - Implementing custom surrogate models
- **[Performance Optimization](performance.md)** - Tips for large-scale problems
- **[Integration Guide](integration.md)** - Using Bgolearn with other tools

### 💡 Examples & Tutorials
- **[Basic Examples](examples/basic_examples.py)** - Fundamental usage patterns
- **[Materials Examples](examples/materials_examples.py)** - Real materials discovery cases
- **[Advanced Examples](examples/advanced_examples.py)** - Complex optimization scenarios
- **[Benchmarks](examples/benchmarks.py)** - Performance comparisons

## 🎓 Learning Paths

### For Beginners
1. [Installation and Setup](getting_started.md#installation)
2. [Basic Optimization Tutorial](getting_started.md#your-first-optimization)
3. [Understanding Acquisition Functions](acquisition_functions.md#introduction)
4. [Simple Visualization](visualization.md#basic-visualization)
5. [Basic Examples](examples/basic_examples.py)

### For Materials Scientists
1. [Materials Discovery Workflow](materials_discovery.md#materials-discovery-workflow)
2. [Composition Optimization](materials_discovery.md#single-objective-optimization)
3. [Multi-Objective Materials Design](materials_discovery.md#multi-objective-optimization)
4. [Constraint Handling](materials_discovery.md#constraint-handling)
5. [Materials Examples](examples/materials_examples.py)

### For Advanced Users
1. [Batch Optimization](batch_optimization.md)
2. [Custom Acquisition Functions](acquisition_functions.md#custom-functions)
3. [Advanced Visualization](visualization.md#interactive-visualizations)
4. [Performance Optimization](performance.md)
5. [Integration with ML Pipelines](integration.md)

## 🚀 Quick Start Example

```python
from Bgolearn import BGOsampling
import pandas as pd
import numpy as np

# Create sample data
X_train = pd.DataFrame(np.random.randn(20, 3), columns=['x1', 'x2', 'x3'])
y_train = pd.Series(np.random.randn(20))
X_candidates = pd.DataFrame(np.random.randn(100, 3), columns=['x1', 'x2', 'x3'])

# Initialize and fit Bgolearn
optimizer = BGOsampling.Bgolearn()
model = optimizer.fit(
    data_matrix=X_train,  # Pass DataFrame directly
    Measured_response=y_train,  # Pass Series directly
    virtual_samples=X_candidates,  # Pass DataFrame directly
    opt_num=1
)

# Get recommendation using Expected Improvement
ei_values, recommended_points = model.EI()
next_experiment = recommended_points[0]
print(f"Next experiment: {next_experiment}")

# Get predictions for all virtual samples
predicted_values = model.virtual_samples_mean
print(f"Number of predictions: {len(predicted_values)}")

# Basic visualization using matplotlib
import matplotlib.pyplot as plt

# Plot EI values
plt.figure(figsize=(10, 6))
plt.plot(ei_values)
plt.title('Expected Improvement Values')
plt.xlabel('Candidate Index')
plt.ylabel('EI Value')
plt.show()
```

## 🔧 Installation

### Basic Installation
```bash
pip install bgolearn
```

### Development Installation
```bash
git clone https://github.com/Bin-Cao/Bgolearn.git
cd Bgolearn
pip install -e .
```

### Optional Dependencies
```bash
# For advanced visualization
pip install plotly seaborn

# For materials science features
pip install pymatgen matminer
```

## 🎯 Key Features

### Acquisition Functions
- **Expected Improvement (EI)** - Balanced exploration/exploitation
- **Upper Confidence Bound (UCB)** - Controllable exploration
- **Probability of Improvement (PoI)** - Conservative optimization
- **Predictive Entropy Search (PES)** - Information-theoretic approach
- **Knowledge Gradient (KG)** - Value of information
- **Batch variants** for parallel experiments

### Visualization Tools
- **Optimization convergence** plots
- **Acquisition function landscapes** (1D/2D)
- **Model uncertainty** visualization
- **Multi-objective Pareto fronts**
- **Interactive dashboards** with Plotly
- **Materials-specific** visualizations

### Materials Science Features
- **Composition optimization** with constraints
- **Processing parameter** optimization
- **Multi-objective** materials design
- **Property trade-off** analysis
- **Experimental design** for synthesis

## 📊 Performance & Scalability

Bgolearn is designed for efficiency:

- **Vectorized computations** using NumPy
- **Batch optimization** for parallel experiments
- **Memory-efficient** candidate handling
- **GPU acceleration** support (future feature)
- **Scalable** to thousands of candidates

## 🤝 Community & Support

### Getting Help
- **Documentation**: Comprehensive guides and examples
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community Q&A and sharing
- **Email**: binjacobcao@gmail.com

### Contributing
We welcome contributions! See our [contribution guidelines](CONTRIBUTING.md) for:
- Code contributions
- Documentation improvements
- Example submissions
- Bug reports and feature requests

### Citation
If you use Bgolearn in your research, please cite:

```bibtex
@article{cao2024bgolearn,
  title={Bgolearn: Bayesian Global Optimization for Materials Design},
  author={Cao, Bin and others},
  journal={Materials \& Design},
  year={2024},
  doi={10.1016/j.matdes.2024.112921}
}
```

## 🗺️ Roadmap

### Current Version (2.4.0)
- ✅ Enhanced acquisition functions
- ✅ Batch optimization
- ✅ Advanced visualization
- ✅ Bug fixes and improvements

### Upcoming Features
- 🔄 GPU acceleration support
- 🔄 Multi-fidelity optimization
- 🔄 Deep Gaussian processes
- 🔄 Cloud platform integration
- 🔄 AutoML capabilities

### Long-term Vision
- 🎯 Standard tool for materials optimization
- 🎯 Integration with major materials databases
- 🎯 Educational platform for optimization
- 🎯 Industry-ready deployment tools

## 📄 License

Bgolearn is released under the MIT License. See [LICENSE](https://github.com/Bin-Cao/Bgolearn/blob/main/LICENSE) for details.

## 🙏 Acknowledgments

Bgolearn builds upon the excellent work of the scientific Python community, including:
- **scikit-learn** for machine learning tools
- **NumPy/SciPy** for numerical computing
- **Matplotlib/Plotly** for visualization
- **Pandas** for data manipulation

Special thanks to all contributors and users who help make Bgolearn better!

---

**Ready to accelerate your research with Bayesian optimization?** Start with our [Getting Started Guide](getting_started.md) or explore the [Examples](examples/) to see Bgolearn in action! 🚀
