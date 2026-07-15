# Figures and Tables Guide for Bgolearn Nature Manuscript

## Main Manuscript Figures

### Figure 1: Framework Architecture
**Type:** Schematic diagram  
**Panels:** 3 panels (A, B, C)

**Panel A - System Architecture:**
- Three-layer architecture diagram
- Data Layer (bottom): Shows experimental data, virtual candidates, constraints
- Surrogate Layer (middle): GP, RF, GB, SVR, MLP models
- Acquisition Layer (top): EI, PI, UCB, EHVI, PES, KG
- Arrows showing data flow between layers

**Panel B - Workflow Diagram:**
- Flowchart showing typical optimization loop:
  1. Initial experiments → Training data
  2. Fit surrogate model
  3. Compute acquisition function
  4. Select next experiment
  5. Perform experiment → Update data
  6. Loop back to step 2
- Highlight decision points and iteration counter

**Panel C - Code Simplicity:**
- Side-by-side comparison:
  - Left: Traditional BO implementation (20-30 lines of complex code)
  - Right: Bgolearn implementation (5 lines)
- Visual emphasis on simplicity

**Dimensions:** Full width (180mm), Height: 120mm  
**File format:** PDF or high-resolution PNG (300 DPI)

---

### Figure 2: Benchmark Performance
**Type:** Multi-panel performance plots  
**Panels:** 4 panels (A, B, C, D)

**Panel A - Branin Function:**
- X-axis: Iteration number (0-50)
- Y-axis: Simple regret (log scale)
- Lines: Bgolearn (EI+GP), Bgolearn (UCB+RF), Random Search, Grid Search
- Shaded regions: 95% confidence intervals (20 runs)
- Highlight convergence point (90% optimal)

**Panel B - Hartmann-6D Function:**
- Same format as Panel A
- Show higher-dimensional performance
- Emphasize Bgolearn's advantage in complex spaces

**Panel C - Acquisition Function Comparison:**
- Heatmap showing acquisition function values over 2D Branin function
- Subplots for EI, UCB, PI
- Overlay: Current observations (dots), next recommended point (star)
- Colorbar indicating acquisition value

**Panel D - Computational Time:**
- Bar chart comparing wall-clock time
- X-axis: Method (Bgolearn-GP, Bgolearn-RF, BoTorch, Ax, Random)
- Y-axis: Time per iteration (seconds, log scale)
- Error bars from multiple runs

**Dimensions:** Full width (180mm), Height: 140mm  
**File format:** PDF with vector graphics

---

### Figure 3: Multi-Objective Optimization Results
**Type:** Pareto front visualization and performance metrics  
**Panels:** 4 panels (A, B, C, D)

**Panel A - 2D Pareto Front (Alloy Design):**
- X-axis: Strength (MPa)
- Y-axis: Ductility (%)
- Points: Initial experiments (gray circles)
- Points: EHVI-discovered solutions (red stars)
- Points: PI-discovered solutions (blue triangles)
- Points: UCB-discovered solutions (green squares)
- Pareto front line connecting non-dominated points
- Annotations for key compositions

**Panel B - 3D Pareto Front:**
- 3D scatter plot
- Axes: Strength, Ductility, Cost
- Color gradient indicating hypervolume contribution
- Rotation angle showing best view of Pareto surface

**Panel C - Hypervolume Convergence:**
- X-axis: Number of experiments
- Y-axis: Hypervolume indicator
- Lines for EHVI, PI, UCB
- Shaded confidence intervals
- Horizontal line: Theoretical maximum (if known)

**Panel D - Algorithm Comparison Matrix:**
- Heatmap showing performance across metrics
- Rows: EHVI, PI, UCB
- Columns: Hypervolume, Pareto points, Computation time, Robustness
- Color scale: Green (best) to Red (worst)
- Numerical values overlaid

**Dimensions:** Full width (180mm), Height: 150mm  
**File format:** PDF with vector graphics

---

### Figure 4: Real Materials Discovery Applications
**Type:** Application showcase  
**Panels:** 3 panels (A, B, C)

**Panel A - Lead-Free Solder Discovery:**
- Left: Ternary composition diagram (Sn-Bi-Zn)
  - Initial experiments (gray dots)
  - BO-discovered alloys (colored by iteration)
  - Optimal region highlighted
- Right: Property evolution plot
  - X-axis: Iteration
  - Y-axis: Strength and Ductility (dual axis)
  - Show improvement trajectory

**Panel B - High-Entropy Alloy Design:**
- Parallel coordinates plot
- Axes: 5 elements (Co, Cr, Fe, Ni, Mn) + 2 properties
- Lines: Each alloy composition
- Color: Pareto rank (red = rank 1, blue = rank 5+)
- Highlight optimal compositions

**Panel C - Ceramic Processing Optimization:**
- 2x2 grid of contour plots
- Variables: Temperature vs Time
- Subplots: Density, Thermal conductivity, Hardness, Cost
- Overlay: Experimental points and BO trajectory
- Star: Final optimized condition

**Dimensions:** Full width (180mm), Height: 120mm  
**File format:** PDF with vector graphics

---

### Figure 5: BgoFace Graphical Interface
**Type:** Software screenshot with annotations  
**Panels:** 2 panels (A, B)

**Panel A - Main Interface:**
- Screenshot of BgoFace main window
- Annotations pointing to key features:
  - Data loading panel
  - Model selection dropdown
  - Acquisition function options
  - Visualization area
  - Results table
- Clean, professional appearance

**Panel B - Workflow Comparison:**
- Timeline diagram comparing:
  - Traditional approach: Literature review → Code development → Debugging → Analysis (2-3 hours)
  - BgoFace approach: Load data → Configure → Run → Analyze (10-15 minutes)
- Visual emphasis on time savings

**Dimensions:** Full width (180mm), Height: 100mm  
**File format:** High-resolution PNG (300 DPI)

---

## Main Manuscript Tables

### Table 1: Acquisition Function Comparison
**Content:**

| Acquisition Function | Formula | Best Use Case | Exploration | Computation |
|---------------------|---------|---------------|-------------|-------------|
| Expected Improvement (EI) | $(f^* - \mu)\Phi(Z) + \sigma\phi(Z)$ | General purpose | Moderate | Low |
| Upper Confidence Bound (UCB) | $\mu + \beta\sigma$ | Noisy functions | High (tunable) | Low |
| Probability of Improvement (PI) | $\Phi((f^* - \mu)/\sigma)$ | Conservative | Low | Low |
| Predictive Entropy Search (PES) | $H[p(x^*\|\mathcal{D})]$ | Information gain | High | High |
| Knowledge Gradient (KG) | $\mathbb{E}[\max \mu_{n+1} - \max \mu_n]$ | Finite budget | Moderate | Medium |
| EHVI (Multi-obj) | $\mathbb{E}[\Delta HV]$ | 2-4 objectives | Moderate | High |

**Dimensions:** Full width  
**Format:** LaTeX booktabs style

---

### Table 2: Surrogate Model Performance
**Content:**

| Model | Training Time | Prediction Time | Uncertainty Quality | Best For |
|-------|---------------|-----------------|---------------------|----------|
| Gaussian Process | O(n³) | O(n) | Excellent | n < 1000, smooth functions |
| Random Forest | O(n log n) | O(log n) | Good (bootstrap) | n > 500, discrete features |
| Gradient Boosting | O(n log n) | O(log n) | Good (bootstrap) | Complex, non-smooth |
| SVR | O(n²) to O(n³) | O(n) | Poor | High-dimensional |
| Neural Network | O(n) per epoch | O(1) | Fair (dropout) | Very large n |

**Dimensions:** Full width  
**Format:** LaTeX booktabs style

---

### Table 3: Materials Discovery Case Studies
**Content:**

| Application | Objectives | Variables | Initial Exp. | BO Exp. | Traditional Est. | Improvement |
|-------------|-----------|-----------|--------------|---------|------------------|-------------|
| Lead-free solder | Strength, Ductility | 3 (Sn, Bi, Zn) | 15 | 28 | 200+ | 82% reduction |
| High-entropy alloy | Strength, Toughness, Cost | 5 elements | 20 | 50 | 300+ | 83% reduction |
| Ceramic processing | Density, Conductivity | 3 (T, t, atm) | 12 | 35 | 150+ | 77% reduction |
| Polymer blend | Modulus, Toughness | 4 components | 18 | 42 | 250+ | 83% reduction |

**Dimensions:** Full width  
**Format:** LaTeX booktabs style

---

## Supplementary Figures

### Supplementary Figure 1: Extended Benchmark Results
- All benchmark functions (Branin, Hartmann-3D, Hartmann-6D, Ackley, Rosenbrock, Levy)
- Multiple acquisition functions
- Multiple surrogate models
- Grid layout: 6 functions × 3 metrics (regret, time, robustness)

### Supplementary Figure 2: Hyperparameter Sensitivity
- Sensitivity analysis for key parameters:
  - GP lengthscale
  - UCB beta parameter
  - RF number of trees
  - Bootstrap iterations
- Show performance degradation with poor choices

### Supplementary Figure 3: Scalability Analysis
- Performance vs dataset size (n = 10 to 10,000)
- Performance vs dimensionality (D = 2 to 50)
- Memory usage
- Computational time

### Supplementary Figure 4: Uncertainty Calibration
- Calibration plots for different surrogate models
- Predicted uncertainty vs actual error
- Reliability diagrams
- Sharpness vs calibration trade-off

### Supplementary Figure 5: Pareto Front Evolution
- Animation frames showing Pareto front discovery over iterations
- 6-8 snapshots from iteration 1 to final
- Show how front improves and expands

### Supplementary Figure 6: Constraint Handling
- Examples of constrained optimization
- Feasible region visualization
- Acquisition function with constraints
- Comparison: penalty method vs constraint-aware acquisition

### Supplementary Figure 7: Batch Optimization
- Batch acquisition strategies
- Parallel experiment selection
- Speedup vs batch size
- Quality vs batch size trade-off

### Supplementary Figure 8: Cross-Validation Results
- Model selection via CV
- CV score vs test performance
- Overfitting detection
- Optimal hyperparameter selection

---

## Supplementary Tables

### Supplementary Table 1: Complete Benchmark Statistics
- Detailed statistics for all benchmark functions
- Mean, median, std, min, max for regret
- Statistical significance tests (t-test, Wilcoxon)

### Supplementary Table 2: Hyperparameter Settings
- Default hyperparameters for all models
- Recommended ranges
- Tuning guidelines

### Supplementary Table 3: Computational Requirements
- Hardware specifications
- Memory usage by dataset size
- Parallelization efficiency
- GPU acceleration (if applicable)

### Supplementary Table 4: Software Comparison
- Feature comparison: Bgolearn vs BoTorch vs Ax vs GPyOpt vs Spearmint
- Ease of use, features, performance, documentation

### Supplementary Table 5: Materials Database
- Complete experimental data for case studies
- Compositions, processing conditions, measured properties
- Metadata (date, operator, equipment)

---

## Figure Preparation Guidelines

### General Requirements:
1. **Resolution:** 300 DPI minimum for raster images
2. **Format:** PDF (vector) preferred, PNG for screenshots
3. **Fonts:** Arial or Helvetica, minimum 8pt
4. **Line width:** Minimum 0.5pt
5. **Colors:** Colorblind-friendly palette (use ColorBrewer)
6. **Size:** Single column (85mm) or full width (180mm)

### Color Palette Recommendations:
- **Primary:** #1f77b4 (blue)
- **Secondary:** #ff7f0e (orange)
- **Tertiary:** #2ca02c (green)
- **Quaternary:** #d62728 (red)
- **Neutral:** #7f7f7f (gray)

### Software Recommendations:
- **Plotting:** Matplotlib, Seaborn (Python)
- **Diagrams:** Inkscape, Adobe Illustrator
- **3D plots:** Mayavi, Plotly
- **Schematics:** draw.io, PowerPoint

---

## Data Visualization Best Practices

1. **Always include error bars or confidence intervals**
2. **Use consistent color schemes across figures**
3. **Label all axes with units**
4. **Include legends for multi-line plots**
5. **Use log scale when data spans orders of magnitude**
6. **Annotate key points and regions**
7. **Keep backgrounds white or light gray**
8. **Avoid 3D pie charts and excessive decoration**
9. **Ensure text is readable when printed in grayscale**
10. **Include scale bars for images**

---

## Figure File Naming Convention

```
fig1_architecture.pdf
fig2_benchmark_performance.pdf
fig3_mobo_results.pdf
fig4_materials_applications.pdf
fig5_bgoface_interface.png
suppfig1_extended_benchmarks.pdf
suppfig2_hyperparameter_sensitivity.pdf
...
```

---

## Table Formatting Example (LaTeX)

```latex
\begin{table}[h]
\centering
\caption{Performance comparison on benchmark functions}
\label{tab:benchmark}
\begin{tabular}{lcccc}
\toprule
Function & Method & Iterations & Regret & Time (s) \\
\midrule
Branin & Bgolearn (EI) & 15 & 0.040 & 2.3 \\
       & Random Search & 42 & 0.089 & 0.5 \\
\midrule
Hartmann-6D & Bgolearn (EI) & 30 & 0.033 & 8.7 \\
            & Random Search & 95 & 0.124 & 1.1 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Checklist Before Submission

- [ ] All figures are high resolution (300 DPI)
- [ ] All text in figures is readable
- [ ] Color schemes are colorblind-friendly
- [ ] All axes are labeled with units
- [ ] All figures have descriptive captions
- [ ] Figure numbers match text references
- [ ] All data points have error bars where appropriate
- [ ] Figures look good in grayscale
- [ ] File sizes are reasonable (<10 MB per figure)
- [ ] All source data is available
- [ ] Figure permissions obtained (if using external images)
- [ ] Consistent style across all figures
- [ ] All abbreviations defined in captions
- [ ] Scale bars included where needed
- [ ] Statistical significance indicated
- [ ] Legends are clear and complete

