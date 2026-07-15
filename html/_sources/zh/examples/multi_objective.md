# 多目标优化示例

```{note}
本页是 Bgolearn 手册的中文版本。
```

```{note}
本页面提供了使用 MultiBgolearn 解决材料设计问题的多目标优化的综合示例。
```

<!-- ```{important}
**MultiBgolearn 限制**：当前版本的 MultiBgolearn **仅支持双目标优化**（`object_num=2`）。
本页中的所有示例均使用 2 个目标。对于超过2个目标，请使用加权求和法
如 Bgolearn 的单目标示例所示。
``` -->

```{warning}
**虚拟空间要求**：
- 虚拟空间（候选点）应包含**足够的样本**以进行优化
- **推荐**：至少100-1000个候选点
- 虚拟空间 CSV 文件应包含**仅特征列**（无目标列）
- 如果您的虚拟空间太小或者有格式问题，您可能会遇到`IndexError`

**示例**：对于 3 特征合金（Cu、Mg、Si），您的 `virtual_space.csv` 应如下所示：
```csv
Cu,Mg,Si
1.5,0.5,0.3
1.6,0.6,0.4
...
```
（虚拟空间中没有强度或延展性柱！）
```

## 示例 1：双目标合金设计

同时优化铝合金的强度和延展性。

### 问题设置

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MultiBgolearn import bgo

# Problem: Optimize Al-Cu-Mg-Si alloy for:
# 1. Maximize Strength (MPa)
# 2. Maximize Ductility (%)
# Note: MultiBgolearn only supports 2 objectives (object_num=2)
```

### 准备训练数据

```python
# Historical experimental data
# IMPORTANT: Only 2 objectives (Strength and Ductility) for MultiBgolearn
dataset = pd.DataFrame({
    'Cu': [2.0, 3.5, 1.8, 4.2, 2.8, 3.2, 2.5, 3.8, 2.2, 3.6],
    'Mg': [1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.9, 1.0, 1.4, 0.7],
    'Si': [0.5, 0.7, 0.3, 0.8, 0.6, 0.4, 0.9, 0.5, 0.7, 0.6],
    'Strength': [250, 280, 240, 290, 265, 275, 255, 285, 245, 275],      # Objective 1
    'Ductility': [15, 12, 18, 10, 14, 13, 16, 11, 17, 12]                # Objective 2
})

print("Training Dataset:")
print(dataset.head())
print(f"\nDataset shape: {dataset.shape}")
print(f"Features: {list(dataset.columns[:3])}")
print(f"Objectives (2 only): {list(dataset.columns[3:])}")

# Save dataset
dataset.to_csv('alloy_dataset.csv', index=False)
```

### 创建虚拟空间

```python
# Generate candidate alloy compositions
# IMPORTANT: Create enough candidates (recommended: 500-2000)
np.random.seed(42)
n_candidates = 1000  # Sufficient number of candidates

# Method 1: Random sampling (recommended for large spaces)
virtual_space = pd.DataFrame({
    'Cu': np.random.uniform(1.5, 4.5, n_candidates),
    'Mg': np.random.uniform(0.5, 1.5, n_candidates),
    'Si': np.random.uniform(0.3, 1.0, n_candidates)
})

# IMPORTANT: Verify virtual space has enough candidates
print(f"\nVirtual space: {len(virtual_space)} candidate compositions")
if len(virtual_space) < 100:
    print("WARNING: Virtual space has fewer than 100 candidates!")
    print("Consider increasing n_candidates for better optimization.")
else:
    print(f"Virtual space size is adequate")

print(f"Virtual space shape: {virtual_space.shape}")
print(f"Virtual space columns: {list(virtual_space.columns)}")
print(f"Composition ranges:")
print(f"  Cu: {virtual_space['Cu'].min():.1f} - {virtual_space['Cu'].max():.1f}%")
print(f"  Mg: {virtual_space['Mg'].min():.1f} - {virtual_space['Mg'].max():.1f}%")
print(f"  Si: {virtual_space['Si'].min():.1f} - {virtual_space['Si'].max():.1f}%")

# Save virtual space (IMPORTANT: only feature columns, no objectives!)
virtual_space.to_csv('virtual_space.csv', index=False)
print("\n Virtual space saved to 'virtual_space.csv'")
```

### 使用 EHVI 进行多目标优化

```python
# Expected Hypervolume Improvement optimization
print("Running EHVI optimization...")

# IMPORTANT: MultiBgolearn only supports bi-objective optimization (object_num=2)
# The fit() method uses positional arguments for the first 3 parameters
VS_recommended, improvements, index = bgo.fit(
    'alloy_dataset.csv',             # dataset_path (positional arg 1)
    'virtual_space.csv',             # VS_path (positional arg 2)
    2,                               # object_num (positional arg 3) - MUST BE 2
    max_search=True,                 # Maximize both objectives
    method='EHVI',                   # Expected Hypervolume Improvement
    assign_model='RandomForest',     # Surrogate model (RandomForest is more stable)
    bootstrap=10                     # Bootstrap iterations for uncertainty
)

print(f"\nEHVI Optimization Results:")
print(f"Recommended composition: {VS_recommended}")
print(f"  Cu: {VS_recommended[0]:.2f}%")
print(f"  Mg: {VS_recommended[1]:.2f}%")
print(f"  Si: {VS_recommended[2]:.2f}%")
print(f"  Total: {sum(VS_recommended):.2f}%")
print(f"\nHypervolume improvement: {improvements[index]:.4f}")
print(f"Recommended index in virtual space: {index}")
```

### 比较不同的 MOBO 算法

```python
# Compare EHVI, PI, and UCB
algorithms = ['EHVI', 'qNEHVI', 'PI', 'UCB']
results = {}

for algorithm in algorithms:
    print(f"\nRunning {algorithm} optimization...")

    VS_rec, imp, idx = bgo.fit(
        'alloy_dataset.csv',             # dataset_path (positional arg 1)
        'virtual_space.csv',             # VS_path (positional arg 2)
        2,                               # object_num (positional arg 3) - MUST BE 2
        max_search=True,
        method=algorithm,
        assign_model='RandomForest',     # Use RandomForest for stability
        bootstrap=8
    )
    
    results[algorithm] = {
        'composition': VS_rec,
        'improvements': imp,
        'index': idx
    }
    
    print(f"{algorithm} recommendation: Cu={VS_rec[0]:.2f}%, Mg={VS_rec[1]:.2f}%, Si={VS_rec[2]:.2f}%")

# Compare results
print("\nAlgorithm Comparison:")
print("-" * 70)
print(f"{'Algorithm':<8} {'Cu (%)':<8} {'Mg (%)':<8} {'Si (%)':<8} {'Total (%)':<10}")
print("-" * 70)
for alg, result in results.items():
    comp = result['composition']
    total = sum(comp)
    print(f"{alg:<8} {comp[0]:<8.2f} {comp[1]:<8.2f} {comp[2]:<8.2f} {total:<10.2f}")
```

### 帕累托前沿分析

```python
# Analyze the Pareto front from training data
def find_pareto_front(objectives):
    """Find Pareto optimal points."""
    n_points = objectives.shape[0]
    pareto_indices = []
    
    for i in range(n_points):
        is_pareto = True
        for j in range(n_points):
            if i != j:
                # Check if point j dominates point i
                if all(objectives[j] >= objectives[i]) and any(objectives[j] > objectives[i]):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    
    return np.array(pareto_indices)

# Extract objectives from training data (bi-objective only!)
objectives = dataset[['Strength', 'Ductility']].values
pareto_indices = find_pareto_front(objectives)
pareto_front = objectives[pareto_indices]

print(f"\nPareto Front Analysis:")
print(f"Found {len(pareto_indices)} Pareto optimal solutions from training data")
print("\nPareto optimal compositions:")
for i, idx in enumerate(pareto_indices):
    comp = dataset.iloc[idx]
    print(f"  {i+1}. Cu={comp['Cu']:.1f}%, Mg={comp['Mg']:.1f}%, Si={comp['Si']:.1f}% -> "
          f"Strength={comp['Strength']:.0f}, Ductility={comp['Ductility']:.0f}")
```

### 可视化

```python
# Create comprehensive visualization for bi-objective optimization
fig = plt.figure(figsize=(15, 10))

# 1. Pareto Front (2D for bi-objective)
ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(objectives[:, 0], objectives[:, 1], alpha=0.6, s=50, c='lightblue', label='All solutions')
ax1.scatter(pareto_front[:, 0], pareto_front[:, 1], c='red', s=100, marker='*', label='Pareto front')

# Add recommended point (estimated values)
rec_objectives = [285, 13]  # Estimated for recommended composition
ax1.scatter(rec_objectives[0], rec_objectives[1], c='gold', s=150, marker='D',
           edgecolors='black', linewidth=2, label='EHVI recommendation')

ax1.set_xlabel('Strength (MPa)')
ax1.set_ylabel('Ductility (%)')
ax1.set_title('Bi-Objective Pareto Front')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Parallel coordinates plot (bi-objective)
ax2 = fig.add_subplot(2, 3, 2)
# Normalize objectives for parallel coordinates
obj_norm = (objectives - objectives.min(axis=0)) / (objectives.max(axis=0) - objectives.min(axis=0))
pareto_norm = (pareto_front - objectives.min(axis=0)) / (objectives.max(axis=0) - objectives.min(axis=0))

x_pos = [0, 1]
obj_names = ['Strength', 'Ductility']

# Plot all solutions
for i in range(len(obj_norm)):
    ax2.plot(x_pos, obj_norm[i], 'b-', alpha=0.3, linewidth=1)

# Highlight Pareto front
for i in range(len(pareto_norm)):
    ax2.plot(x_pos, pareto_norm[i], 'r-', alpha=0.8, linewidth=2)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(obj_names)
ax2.set_ylabel('Normalized Value')
ax2.set_title('Parallel Coordinates (Bi-Objective)')
ax2.grid(True, alpha=0.3)

# 3. Algorithm comparison
ax3 = fig.add_subplot(2, 3, 3)
algorithms = list(results.keys())
compositions = [results[alg]['composition'] for alg in algorithms]
cu_values = [comp[0] for comp in compositions]
mg_values = [comp[1] for comp in compositions]
si_values = [comp[2] for comp in compositions]

x = np.arange(len(algorithms))
width = 0.25

ax3.bar(x - width, cu_values, width, label='Cu', alpha=0.8)
ax3.bar(x, mg_values, width, label='Mg', alpha=0.8)
ax3.bar(x + width, si_values, width, label='Si', alpha=0.8)

ax3.set_xlabel('Algorithm')
ax3.set_ylabel('Composition (%)')
ax3.set_title('Algorithm Recommendations')
ax3.set_xticks(x)
ax3.set_xticklabels(algorithms)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Objective distribution
ax4 = fig.add_subplot(2, 3, 4)
ax4.hist(objectives[:, 0], bins=15, alpha=0.6, label='Strength', color='blue')
ax4.axvline(pareto_front[:, 0].mean(), color='red', linestyle='--', linewidth=2, label='Pareto mean')
ax4.set_xlabel('Strength (MPa)')
ax4.set_ylabel('Frequency')
ax4.set_title('Strength Distribution')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Ductility distribution
ax5 = fig.add_subplot(2, 3, 5)
ax5.hist(objectives[:, 1], bins=15, alpha=0.6, label='Ductility', color='green')
ax5.axvline(pareto_front[:, 1].mean(), color='red', linestyle='--', linewidth=2, label='Pareto mean')
ax5.set_xlabel('Ductility (%)')
ax5.set_ylabel('Frequency')
ax5.set_title('Ductility Distribution')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Composition space
ax6 = fig.add_subplot(2, 3, 6)
scatter = ax6.scatter(dataset['Cu'], dataset['Mg'],
                     c=dataset['Strength'], s=dataset['Ductility']*10,
                     cmap='viridis', alpha=0.6, edgecolors='black')
ax6.set_xlabel('Cu (%)')
ax6.set_ylabel('Mg (%)')
ax6.set_title('Composition Space (color=Strength, size=Ductility)')
plt.colorbar(scatter, ax=ax6, label='Strength (MPa)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 示例 2：双目标处理优化

优化硬度和韧性的热处理参数。

### 问题设置

```python
# Heat treatment optimization for:
# 1. Maximize Hardness (HV)
# 2. Maximize Toughness (J)
# Note: MultiBgolearn only supports 2 objectives (object_num=2)

processing_dataset = pd.DataFrame({
    'Temperature': [450, 500, 550, 480, 520, 470, 530, 490, 510, 460],
    'Time': [2, 4, 6, 3, 5, 2.5, 4.5, 3.5, 4.2, 2.8],
    'Cooling_Rate': [10, 20, 15, 25, 12, 18, 22, 14, 16, 19],
    'Hardness': [180, 220, 250, 200, 235, 190, 245, 210, 230, 185],      # Objective 1
    'Toughness': [45, 35, 25, 40, 30, 42, 28, 38, 32, 44]                # Objective 2
})

print("Processing Dataset:")
print(processing_dataset.head())
print(f"Objectives (2 only): Hardness, Toughness")

# Save dataset
processing_dataset.to_csv('processing_dataset.csv', index=False)

# Create virtual processing space
# IMPORTANT: Generate enough candidates (recommended: 500-2000)
np.random.seed(123)
n_conditions = 800  # Sufficient number of candidate conditions

virtual_processing_df = pd.DataFrame({
    'Temperature': np.random.uniform(440, 560, n_conditions),
    'Time': np.random.uniform(1.5, 6.5, n_conditions),
    'Cooling_Rate': np.random.uniform(8, 28, n_conditions)
})

virtual_processing_df.to_csv('virtual_processing.csv', index=False)

print(f"Virtual processing space: {len(virtual_processing_df)} conditions")
print(f"Virtual space size is adequate")
```

### 双目标优化

```python
# Optimize for 2 objectives (Hardness and Toughness)
# IMPORTANT: object_num MUST be 2 for MultiBgolearn
VS_recommended, improvements, index = bgo.fit(
    'processing_dataset.csv',        # dataset_path (positional arg 1)
    'virtual_processing.csv',        # VS_path (positional arg 2)
    2,                               # object_num (positional arg 3) - MUST BE 2
    max_search=True,                 # Maximize both objectives
    method='EHVI',                   # Expected Hypervolume Improvement
    assign_model='RandomForest',     # Use RandomForest for stability
    bootstrap=10
)

print(f"\nBi-Objective Optimization Results:")
print(f"Recommended processing conditions:")
print(f"  Temperature: {VS_recommended[0]:.0f}°C")
print(f"  Time: {VS_recommended[1]:.1f} hours")
print(f"  Cooling Rate: {VS_recommended[2]:.0f}°C/min")

print(f"\nExpected improvements for 2 objectives:")
print(f"  Hardness improvement: {improvements[0]:.1f} HV")
print(f"  Toughness improvement: {improvements[1]:.1f} J")
```

## 示例 3：有约束的材料发现

在约束条件下优化陶瓷成分。

### 问题设置

```python
# Ceramic optimization for:
# 1. Maximize Strength (MPa)
# 2. Maximize Thermal Conductivity (W/mK)
# Note: MultiBgolearn only supports 2 objectives (object_num=2)

ceramic_dataset = pd.DataFrame({
    'Al2O3': [85, 90, 80, 95, 88, 92, 83, 89, 87, 91],
    'SiO2': [10, 5, 15, 3, 8, 4, 12, 7, 9, 6],
    'MgO': [5, 5, 5, 2, 4, 4, 5, 4, 4, 3],
    'Strength': [300, 350, 280, 380, 320, 360, 290, 340, 330, 365],              # Objective 1
    'Thermal_Conductivity': [25, 30, 20, 35, 28, 32, 22, 29, 27, 33]            # Objective 2
})

print("Ceramic Dataset:")
print(ceramic_dataset.head())
print(f"Objectives (2 only): Strength, Thermal_Conductivity")

# Constraint: compositions must sum to 100%
def check_ceramic_constraints(composition):
    """Check ceramic composition constraints."""
    al2o3, sio2, mgo = composition
    
    # Composition sum constraint (within tolerance)
    total = al2o3 + sio2 + mgo
    if not (99 <= total <= 101):
        return False
    
    # Individual component constraints
    if not (75 <= al2o3 <= 98):
        return False
    if not (2 <= sio2 <= 20):
        return False
    if not (1 <= mgo <= 8):
        return False
    
    return True

# Generate constrained virtual space
# IMPORTANT: Create enough candidates (recommended: 500-2000)
np.random.seed(456)
n_ceramic_candidates = 1000

virtual_ceramics = []
for _ in range(n_ceramic_candidates * 2):  # Generate extra to account for constraint filtering
    al2o3 = np.random.uniform(75, 98)
    sio2 = np.random.uniform(2, 20)
    mgo = np.random.uniform(1, 8)

    # Normalize to sum to 100%
    total = al2o3 + sio2 + mgo
    normalized = [al2o3 * 100/total, sio2 * 100/total, mgo * 100/total]

    if check_ceramic_constraints(normalized):
        virtual_ceramics.append(normalized)

    if len(virtual_ceramics) >= n_ceramic_candidates:
        break

virtual_ceramics_df = pd.DataFrame(virtual_ceramics, columns=['Al2O3', 'SiO2', 'MgO'])
print(f"Constrained virtual ceramic space: {len(virtual_ceramics_df)} compositions")
if len(virtual_ceramics_df) >= 500:
    print(f"Virtual space size is adequate")
else:
    print(f"WARNING: Only {len(virtual_ceramics_df)} candidates generated")

# Save data
ceramic_dataset.to_csv('ceramic_dataset.csv', index=False)
virtual_ceramics_df.to_csv('virtual_ceramics.csv', index=False)
```

### 约束多目标优化

```python
# Bi-objective ceramic optimization
# IMPORTANT: object_num MUST be 2 for MultiBgolearn
VS_recommended, improvements, index = bgo.fit(
    'ceramic_dataset.csv',           # dataset_path (positional arg 1)
    'virtual_ceramics.csv',          # VS_path (positional arg 2)
    2,                               # object_num (positional arg 3) - MUST BE 2
    max_search=True,
    method='EHVI',
    assign_model='RandomForest',     # Use RandomForest for stability
    bootstrap=8
)

print(f"Ceramic Optimization Results:")
print(f"Recommended composition:")
print(f"  Al2O3: {VS_recommended[0]:.1f}%")
print(f"  SiO2: {VS_recommended[1]:.1f}%")
print(f"  MgO: {VS_recommended[2]:.1f}%")
print(f"  Total: {sum(VS_recommended):.1f}%")

# Verify constraints
if check_ceramic_constraints(VS_recommended):
    print("✓ All constraints satisfied")
else:
    print("✗ Constraint violation detected")

print(f"Expected improvements:")
print(f"  Strength: +{improvements[0]:.1f} MPa")
print(f"  Thermal Conductivity: +{improvements[1]:.1f} W/mK")
print(f"  Thermal Shock Resistance: +{improvements[2]:.1f} cycles")
```

## 示例4：敏感性分析

分析优化对不同因素的敏感度。

### 模型灵敏度

```python
# Compare different surrogate models for multi-objective optimization
# Note: RandomForest and GradientBoosting are generally more stable than GaussianProcess
models_to_test = ['RandomForest', 'GradientBoosting', 'LinearRegression']
model_results = {}

for model_name in models_to_test:
    print(f"\nTesting {model_name} for multi-objective optimization...")

    try:
        VS_rec, imp, idx = bgo.fit(
            'alloy_dataset.csv',             # dataset_path (positional arg 1)
            'virtual_space.csv',             # VS_path (positional arg 2)
            2,                               # object_num (positional arg 3) - MUST BE 2
            max_search=True,
            method='EHVI',
            assign_model=model_name,
            bootstrap=5  # Reduced for speed
        )
        
        model_results[model_name] = {
            'composition': VS_rec,
            'improvements': imp,
            'success': True
        }
        
        print(f"Success: Cu={VS_rec[0]:.2f}%, Mg={VS_rec[1]:.2f}%, Si={VS_rec[2]:.2f}%")
        
    except Exception as e:
        print(f"Failed: {str(e)}")
        model_results[model_name] = {'success': False, 'error': str(e)}

# Compare successful results
print("\nModel Comparison for Multi-Objective Optimization:")
print("-" * 80)
print(f"{'Model':<15} {'Cu (%)':<8} {'Mg (%)':<8} {'Si (%)':<8} {'Improvements':<20}")
print("-" * 80)

for model, result in model_results.items():
    if result['success']:
        comp = result['composition']
        imp = result['improvements']
        imp_str = f"[{imp[0]:.1f}, {imp[1]:.1f}, {imp[2]:.1f}]"
        print(f"{model:<15} {comp[0]:<8.2f} {comp[1]:<8.2f} {comp[2]:<8.2f} {imp_str:<20}")
    else:
        print(f"{model:<15} {'Failed':<40}")
```

### 自举灵敏度

```python
# Test different bootstrap settings
bootstrap_values = [3, 5, 8, 10, 15]
bootstrap_results = {}

for bootstrap in bootstrap_values:
    print(f"\nTesting bootstrap = {bootstrap}...")

    VS_rec, imp, idx = bgo.fit(
        'alloy_dataset.csv',             # dataset_path (positional arg 1)
        'virtual_space.csv',             # VS_path (positional arg 2)
        2,                               # object_num (positional arg 3) - MUST BE 2
        max_search=True,
        method='EHVI',
        assign_model='RandomForest',     # Use RandomForest for stability
        bootstrap=bootstrap
    )
    
    bootstrap_results[bootstrap] = {
        'composition': VS_rec,
        'improvements': imp
    }

# Analyze bootstrap sensitivity
print("\nBootstrap Sensitivity Analysis:")
print("-" * 70)
print(f"{'Bootstrap':<10} {'Cu (%)':<8} {'Mg (%)':<8} {'Si (%)':<8} {'Avg Improvement':<15}")
print("-" * 70)

for bootstrap, result in bootstrap_results.items():
    comp = result['composition']
    avg_imp = np.mean(result['improvements'])
    print(f"{bootstrap:<10} {comp[0]:<8.2f} {comp[1]:<8.2f} {comp[2]:<8.2f} {avg_imp:<15.2f}")
```

## 多目标优化的最佳实践

### 1. 问题表述
```python
# Good practice: Clear objective definitions
objectives = {
    'Strength': {'type': 'maximize', 'unit': 'MPa', 'range': [200, 350]},
    'Ductility': {'type': 'maximize', 'unit': '%', 'range': [8, 20]},
    'Cost': {'type': 'minimize', 'unit': '$/kg', 'range': [80, 150]}
}

# Convert minimization to maximization
for obj_name, obj_info in objectives.items():
    if obj_info['type'] == 'minimize':
        print(f"Converting {obj_name} to maximization (negative values)")
```

### 2. 数据预处理
```python
# Normalize objectives if they have very different scales
from sklearn.preprocessing import StandardScaler

def preprocess_objectives(data, objective_columns):
    """Preprocess objectives for multi-objective optimization."""
    scaler = StandardScaler()
    normalized_data = data.copy()
    normalized_data[objective_columns] = scaler.fit_transform(data[objective_columns])
    return normalized_data, scaler

# Example usage
# normalized_dataset, scaler = preprocess_objectives(dataset, ['Strength', 'Ductility', 'Cost'])
```

### 3. 算法选择指南
```python
algorithm_guidelines = {
    'EHVI': {
        'best_for': '2-4 objectives',
        'pros': ['Theoretically sound', 'Balanced exploration'],
        'cons': ['Computationally expensive for >4 objectives'],
        'recommended_bootstrap': 8
    },
    'PI': {
        'best_for': 'Conservative optimization',
        'pros': ['Fast computation', 'Reliable improvements'],
        'cons': ['May not explore enough'],
        'recommended_bootstrap': 5
    },
    'UCB': {
        'best_for': 'Noisy objectives, >4 objectives',
        'pros': ['Uncertainty aware', 'Scalable'],
        'cons': ['Requires parameter tuning'],
        'recommended_bootstrap': 10
    }
}

# Print guidelines
for alg, info in algorithm_guidelines.items():
    print(f"\n{alg}:")
    print(f"  Best for: {info['best_for']}")
    print(f"  Recommended bootstrap: {info['recommended_bootstrap']}")
```

## 故障排除

### 常见错误：索引错误

**错误信息**：
```
IndexError: index XXXX is out of bounds for axis 0 with size YYYY
```

**原因**：当推荐的索引超出您的虚拟空间大小时，会出现此错误。

**解决方案**：

1. **检查虚拟空间大小**：
   ```python
   import pandas as pd
   vs = pd.read_csv('virtual_space.csv')
   print(f"Virtual space size: {len(vs)}")
   print(f"Virtual space columns: {list(vs.columns)}")
   ```

2. **确保有足够的候选人**：
   - 建议：至少100-1000候选点
   - 如果您的积分<100，请扩展您的虚拟空间

3. **验证文件格式**：
   ```python
   # Virtual space should ONLY have feature columns
   # CORRECT:
   virtual_space = pd.DataFrame({
       'Cu': [...],
       'Mg': [...],
       'Si': [...]
   })

   # WRONG (includes objectives):
   virtual_space = pd.DataFrame({
       'Cu': [...],
       'Mg': [...],
       'Si': [...],
       'Strength': [...],  # Remove this!
       'Ductility': [...]  # Remove this!
   })
   ```

4. **检查数据是否损坏**：
   ```python
   # Verify no NaN or infinite values
   vs = pd.read_csv('virtual_space.csv')
   print(f"NaN values: {vs.isna().sum().sum()}")
   print(f"Infinite values: {np.isinf(vs.values).sum()}")
   ```

### 常见错误：GaussianProcess 的类型错误

**错误信息**：
```
TypeError: predict() got an unexpected keyword argument 'return_std'
```

**原因**：这是旧版本 MultiBgolearn 中的错误。

**解决方案**：更新到最新版本：
```bash
pip install --upgrade MultiBgolearn
```

或者使用不同的代理模型：
```python
VS_recommended, improvements, index = bgo.fit(
    'dataset.csv',
    'virtual_space.csv',
    2,
    max_search=True,
    method='EHVI',
    assign_model='RandomForest',  # Use RandomForest instead
    bootstrap=10
)
```

## 下一步

- **学习帕累托分析**：{doc}`../pareto_optimization`
- **了解MOBO理论**：{doc}`../multi_objective_concepts`
