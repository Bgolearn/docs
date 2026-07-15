# Basic Concepts

```{note}
本ページは Bgolearn マニュアルの日本語版です。
```

```{note}
This page introduces the fundamental concepts of Bayesian optimization and how Bgolearn implements them.
```

## What is Bayesian Optimization?

Bayesian optimization is a powerful technique for optimizing expensive-to-evaluate functions. It's particularly useful when:

- **Function evaluations are costly** (experiments, simulations)
- **Derivatives are unavailable** (black-box functions)
- **Noise is present** in measurements
- **Few evaluations are possible** (limited budget)

```{admonition} Key Idea
:class: tip
Instead of randomly sampling or using grid search, Bayesian optimization builds a probabilistic model of the function and uses it to intelligently decide where to sample next.
```

## Core Components

### 1. Surrogate Model

The surrogate model approximates the expensive function using previous observations.

**Gaussian Process (GP)** is the most common choice:
- Provides both **mean prediction** and **uncertainty estimate**
- Naturally handles noise in observations
- Flexible and well-suited for many problems

```python
# Example: Fitting a GP model in Bgolearn
from Bgolearn import BGOsampling

optimizer = BGOsampling.Bgolearn()
model = optimizer.fit(
    data_matrix=X_train,
    Measured_response=y_train,
    virtual_samples=X_candidates
)

# Get predictions with uncertainty
mean_pred = model.virtual_samples_mean
std_pred = model.virtual_samples_std
```

### 2. Acquisition Function

The acquisition function decides where to sample next by balancing:
- **Exploitation**: Sample where the model predicts good values
- **Exploration**: Sample where uncertainty is high

Common acquisition functions in Bgolearn:

```{list-table} Acquisition Functions
:header-rows: 1
:name: acquisition-functions-table

* - Function
  - Description
  - Best For
* - **EI** (Expected Improvement)
  - Expected improvement over current best
  - General purpose, balanced exploration/exploitation
* - **UCB** (Upper Confidence Bound)
  - Optimistic estimate with confidence
  - Noisy functions, exploration-focused
* - **PI** (Probability of Improvement)
  - Probability of improving current best
  - Conservative, exploitation-focused
* - **PES** (Predictive Entropy Search)
  - Information-theoretic approach
  - Complex functions, limited budget
```

### 3. Optimization Loop

The Bayesian optimization process follows this iterative loop:

```
1. Initial Data
   ↓
2. Fit Surrogate Model
   ↓
3. Optimize Acquisition Function
   ↓
4. Evaluate at New Point
   ↓
5. Update Dataset
   ↓
6. Stopping Criterion?
   ├─ No → Go back to step 2
   └─ Yes → Return Best Solution
```

## Mathematical Foundation

### Gaussian Process

A Gaussian Process is defined by:
- **Mean function**: $m(x) = \mathbb{E}[f(x)]$
- **Covariance function**: $k(x, x') = \text{Cov}[f(x), f(x')]$

For any finite set of points, the function values follow a multivariate Gaussian distribution:

$$f(x_1), \ldots, f(x_n) \sim \mathcal{N}(\mu, K)$$

where $\mu_i = m(x_i)$ and $K_{ij} = k(x_i, x_j)$.

### Expected Improvement

The Expected Improvement acquisition function is:

$$\text{EI}(x) = \mathbb{E}[\max(f(x) - f^*, 0)]$$

where $f^*$ is the current best observed value.

For a GP posterior with mean $\mu(x)$ and variance $\sigma^2(x)$:

$$\text{EI}(x) = (\mu(x) - f^*)\Phi(Z) + \sigma(x)\phi(Z)$$

where $Z = \frac{\mu(x) - f^*}{\sigma(x)}$, $\Phi$ is the CDF, and $\phi$ is the PDF of the standard normal distribution.

## Practical Considerations

### When to Use Bayesian Optimization

✅ **Good for:**
- Expensive function evaluations (>1 second per evaluation)
- Continuous or mixed-variable spaces
- Noisy observations
- Limited evaluation budget (10-1000 evaluations)
- Black-box functions without derivatives

❌ **Not ideal for:**
- Very cheap functions (use gradient-based methods)
- Very high dimensions (>20 variables)
- Discrete combinatorial problems
- Functions with known structure

### Choosing Acquisition Functions

```{admonition} Quick Guide
:class: tip

- **Start with EI**: Good general-purpose choice
- **Use UCB for noisy functions**: Better exploration
- **Try PI for exploitation**: When you want to be conservative
- **Consider PES for complex functions**: Information-theoretic approach
```

### Handling Constraints

Bgolearn supports various constraint types:

1. **Box constraints**: Simple bounds on variables
2. **Linear constraints**: Linear equality/inequality constraints
3. **Nonlinear constraints**: General constraint functions
4. **Categorical variables**: Discrete choices

```python
# Example: Box constraints
bounds = {
    'temperature': (100, 500),  # Temperature range
    'pressure': (1, 10),        # Pressure range
    'composition': (0, 1)       # Composition fraction
}
```

## Next Steps

Now that you understand the basics:

1. **Learn about acquisition functions**: {doc}`acquisition_functions`


```{seealso}
For deeper mathematical background, see:
- `Mockus et al., The application of Bayesian methods for seeking the extremum` for Gaussian Processes
- `Zhan et al.,Expected improvement for expensive optimization: a review` for the original EI paper

```
