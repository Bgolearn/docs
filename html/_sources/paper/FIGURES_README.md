# Bgolearn Manuscript Figures

This directory contains all figures for the Bgolearn Nature manuscript. Each figure is saved as both PDF (vector, for publication) and PNG (raster, for preview).

## 📊 Generated Figures

### 1. Convergence Plots (Performance vs Iteration)

These show how optimization performance improves over iterations:

- **`convergence_hartmann6d.pdf/png`**
  - Single-objective optimization on Hartmann-6D benchmark
  - Y-axis: Simple regret (log scale)
  - X-axis: Iteration number (0-50)
  - Shows: Bgolearn-GP, Bgolearn-RF, Latin Hypercube, Random Search
  - **Lines**: Mean over 30 independent runs
  - **Shaded regions**: ±1 standard deviation
  - **Use for**: Demonstrating fast convergence of Bayesian optimization with statistical confidence

- **`convergence_ackley5d.pdf/png`**
  - Single-objective optimization on Ackley-5D benchmark
  - Y-axis: Simple regret (log scale)
  - X-axis: Iteration number (0-50)
  - Shows: Same methods as above
  - **Lines**: Mean over 30 independent runs
  - **Shaded regions**: ±1 standard deviation
  - **Use for**: Showing performance on highly multimodal functions with uncertainty quantification

- **`convergence_zdt1_hypervolume.pdf/png`**
  - Multi-objective optimization on ZDT1 benchmark
  - Y-axis: Hypervolume indicator (0-1)
  - X-axis: Iteration number (0-50)
  - Shows: Bgolearn-EHVI, Bgolearn-MO-UCB, NSGA-II, Latin Hypercube
  - **Lines**: Mean over 20 independent runs
  - **Shaded regions**: ±1 standard deviation
  - **Use for**: Demonstrating multi-objective Pareto front improvement with variability

- **`convergence_dtlz2_hypervolume.pdf/png`**
  - Multi-objective optimization on DTLZ2 benchmark (3 objectives)
  - Y-axis: Hypervolume indicator (0-1)
  - X-axis: Iteration number (0-50)
  - Shows: Same methods as ZDT1
  - **Lines**: Mean over 20 independent runs
  - **Shaded regions**: ±1 standard deviation
  - **Use for**: Showing scalability to 3+ objectives with statistical robustness

### 2. Bar Charts (Final Performance Comparison)

These show the final benchmark results with error bars:

- **`bar_hartmann6d.pdf/png`**
  - Iterations to 90% optimality on Hartmann-6D
  - 4 methods: Random Search, Latin Hypercube, Bgolearn-GP, Bgolearn-RF
  - Error bars show standard deviation (n=30 runs)
  - **Use for**: Quantitative comparison of sample efficiency

- **`bar_ackley5d.pdf/png`**
  - Iterations to 90% optimality on Ackley-5D
  - Same 4 methods as above
  - **Use for**: Performance on multimodal problems

- **`bar_zdt1.pdf/png`**
  - Hypervolume indicator on ZDT1 (bi-objective)
  - 5 methods: Random, LHS, NSGA-II, Bgolearn-EHVI, Bgolearn-MO-UCB
  - Error bars show standard deviation (n=20 runs)
  - **Use for**: Multi-objective performance comparison

- **`bar_dtlz2.pdf/png`**
  - Hypervolume indicator on DTLZ2 (tri-objective)
  - Same 5 methods as ZDT1
  - **Use for**: Scalability to higher-dimensional objective spaces

### 3. Computational Efficiency

- **`computational_time.pdf/png`**
  - Time per iteration (seconds) for all methods
  - Grouped bar chart: Single-objective vs Multi-objective
  - Y-axis: Log scale (seconds)
  - **Use for**: Showing computational cost trade-offs

### 4. Combined Figures (For Reference)

These are multi-panel figures combining individual plots:

- **`figure1_single_objective.pdf/png`**
  - 2-panel: Hartmann-6D + Ackley-5D bar charts
  - Size: 7.2" × 2.5" (Nature double-column width)

- **`figure2_multi_objective.pdf/png`**
  - 2-panel: ZDT1 + DTLZ2 bar charts
  - Size: 7.2" × 2.5"

- **`figure3_combined_overview.pdf/png`**
  - 4-panel: All benchmark bar charts
  - Size: 7.2" × 5.5"
  - **Recommended for main manuscript**

## 🎨 Figure Style

All figures follow Nature journal guidelines:

- **Font**: Arial/Helvetica, 9-10pt
- **Colors**: Colorblind-friendly palette
  - Blue (#0173B2): Bgolearn-GP/EHVI (primary method)
  - Orange (#DE8F05): Bgolearn-RF/MO-UCB (secondary method)
  - Green (#029E73): NSGA-II (evolutionary baseline)
  - Purple (#CC78BC): Latin Hypercube (space-filling baseline)
  - Gray (#949494): Random Search (random baseline)
- **Line width**: 2.0pt for curves, 1.0pt for axes
- **DPI**: 300 (publication quality)
- **Format**: PDF (vector) + PNG (raster)

## 📐 Recommended Usage

### For Main Manuscript

1. **Figure 1**: Use `figure3_combined_overview.pdf` (4-panel overview)
   - Shows all benchmark results in one figure
   - Fits Nature double-column format

2. **Figure 2**: Use convergence plots
   - Panel A: `convergence_hartmann6d.pdf`
   - Panel B: `convergence_zdt1_hypervolume.pdf`
   - Shows optimization dynamics over time

3. **Figure 3**: Use `computational_time.pdf`
   - Shows efficiency trade-offs

### For Supplementary Information

- All individual bar charts
- Additional convergence plots (Ackley, DTLZ2)
- Detailed method comparisons

### For Presentations

- Use PNG versions (easier to embed)
- Individual plots allow flexible layout
- High resolution (300 DPI) suitable for projection

## 🔧 Customization

To modify figures, edit the Python scripts:

- **`plot_individual_figures.py`**: Individual plots (convergence, bars, efficiency)
- **`plot_benchmark_comparison.py`**: Combined multi-panel figures

### Common Modifications

1. **Change colors**: Edit `COLORS` dictionary
2. **Adjust figure size**: Modify `figsize=(width, height)` in inches
3. **Update data**: Edit data dictionaries at top of scripts
4. **Font sizes**: Modify `plt.rcParams` settings
5. **Add/remove methods**: Edit `methods` dictionaries in plot functions

### Regenerate Figures

```bash
cd paper
python plot_individual_figures.py      # Individual plots
python plot_benchmark_comparison.py    # Combined figures
```

## 📏 Figure Dimensions

Nature journal specifications:

- **Single column**: 3.5" (89 mm) width
- **Double column**: 7.2" (183 mm) width
- **Maximum height**: 9.0" (229 mm)

Current figures:
- Individual plots: 3.5-4.5" × 3" (suitable for single column)
- Combined figures: 7.2" × 2.5-5.5" (suitable for double column)

## ✅ Quality Checklist

Before submission, verify:

- [ ] All text is readable at 100% zoom
- [ ] Error bars are visible
- [ ] Legend does not overlap data
- [ ] Axis labels are clear and bold
- [ ] Colors are distinguishable in grayscale
- [ ] PDF files are vector format (scalable)
- [ ] File sizes are reasonable (<500 KB per figure)

## 📝 Figure Captions (Draft)

### Figure 1: Benchmark Performance Overview
**Benchmark optimization performance comparison.** (**a**) Single-objective optimization on Hartmann-6D showing iterations to 90% optimality. (**b**) Single-objective optimization on Ackley-5D. (**c**) Multi-objective optimization on ZDT1 showing hypervolume indicator. (**d**) Multi-objective optimization on DTLZ2. Error bars represent standard deviation over 30 runs (single-objective) or 20 runs (multi-objective). Bgolearn-GP/EHVI achieves 3.2× faster convergence than random search and 2.4× faster than Latin hypercube sampling on average.

### Figure 2: Convergence Dynamics
**Optimization convergence over iterations.** (**a**) Simple regret reduction on Hartmann-6D benchmark (mean over n=30 runs). (**b**) Hypervolume growth on ZDT1 benchmark (mean over n=20 runs). Bgolearn methods (solid and dashed lines) converge significantly faster than baseline methods (dotted lines). Shaded regions show ±1 standard deviation, demonstrating consistent performance across multiple independent runs.

### Figure 3: Computational Efficiency
**Computational time per iteration.** Comparison of wall-clock time for single-objective (blue) and multi-objective (orange) optimization. Bgolearn-RF/MO-UCB offers 85% of GP/EHVI performance at 60% of computational cost. Y-axis in log scale.

---

**Generated by**: `plot_individual_figures.py` and `plot_benchmark_comparison.py`  
**Date**: 2024-10-28  
**For**: Bgolearn Nature manuscript

