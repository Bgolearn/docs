# 优化策略

```{note}
本页是 Bgolearn 手册的中文版本。
```

```{note}
本页介绍了充分利用 Bgolearn 的高级优化策略和最佳实践。
```

## 概述

使用 Bgolearn 进行有效优化不仅需要了解算法，还需要了解将其应用于实际问题的策略。本页介绍了先进技术、最佳实践以及需要避免的常见陷阱。

## 顺序优化与批量优化

### 顺序优化

一次进行一个实验的传统方法：

```python
from Bgolearn import BGOsampling

# Sequential optimization loop
opt = BGOsampling.Bgolearn()

for iteration in range(10):
    # Fit model and get recommendation
    model = opt.fit(
        data_matrix=current_data,
        Measured_response=current_response,
        virtual_samples=virtual_samples,
        opt_num=1  # Single recommendation
    )
    
    # Conduct experiment
    ei_values, recommended_points = model.EI()
    new_x = recommended_points[0]
    new_y = conduct_experiment(new_x)
    
    # Update dataset
    current_data = np.vstack([current_data, new_x])
    current_response = np.append(current_response, new_y)
```

**优点：**
- 每个实验的最大信息增益
- 适应新信息
- 降低每次迭代的计算成本

**缺点：**
- 整体进度较慢
- 无法利用并行实验能力
- 每个实验的开销较高

### 批量优化

同时选择多个实验：

```python
# Batch optimization
model = opt.fit(
    data_matrix=current_data,
    Measured_response=current_response,
    virtual_samples=virtual_samples,
    opt_num=5  # Multiple recommendations
)

# Get batch of experiments
batch_indices = model['recommended_indices']
batch_experiments = virtual_samples[batch_indices]

# Conduct experiments in parallel
batch_results = parallel_experiments(batch_experiments)
```

**优点：**
- 整体进步更快
- 利用并行实验能力
- 更好的资源利用率

**缺点：**
- 批次内适应性较差
- 更高的计算成本
- 批次中潜在的冗余

## 探索与利用策略

### 以探索为重点的策略

当您需要彻底探索设计空间时：

```python
# High exploration settings
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Classifier='GaussianProcess',  # Good uncertainty estimates
    # Use acquisition functions that favor exploration
    Dynamic_W=True,  # Dynamic weighting
    seed=42
)
```

**使用时间：**
- 优化的早期阶段
- 巨大的、未开发的设计空间
- 测量不确定度高
- 以发现为中心的研究

### 以剥削为重点的战略

当您想要细化已知的良好区域时：

```python
# High exploitation settings
# Focus virtual space around known good regions
good_region_mask = (virtual_samples[:, 0] > best_composition[0] - 0.5) & \
                   (virtual_samples[:, 0] < best_composition[0] + 0.5)
focused_virtual = virtual_samples[good_region_mask]

model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=focused_virtual,
    opt_num=3  # Multiple points in good region
)
```

**使用时间：**
- 优化后期阶段
- 易于理解的系统
- 实验预算有限
- 以优化为重点的研究

## 适应性策略

### 动态虚拟空间

根据优化进度调整虚拟空间：

```python
def adaptive_virtual_space(iteration, best_point, initial_space):
    """Adapt virtual space based on optimization progress."""
    if iteration < 5:
        # Early: broad exploration
        return initial_space
    elif iteration < 15:
        # Middle: focus around promising regions
        distances = np.linalg.norm(initial_space - best_point, axis=1)
        close_mask = distances < np.percentile(distances, 50)
        return initial_space[close_mask]
    else:
        # Late: local refinement
        distances = np.linalg.norm(initial_space - best_point, axis=1)
        close_mask = distances < np.percentile(distances, 25)
        return initial_space[close_mask]

# Use in optimization loop
for iteration in range(20):
    current_virtual = adaptive_virtual_space(iteration, best_so_far, full_virtual_space)
    model = opt.fit(virtual_samples=current_virtual, ...)
```

### 多阶段优化

不同阶段不同策略：

```python
def multi_stage_optimization(data_matrix, measured_response, virtual_samples):
    """Multi-stage optimization strategy."""
    
    # Stage 1: Broad exploration (Random Forest for robustness)
    print("Stage 1: Exploration")
    for i in range(5):
        model = opt.fit(
            data_matrix=data_matrix,
            Measured_response=measured_response,
            virtual_samples=virtual_samples,
            Classifier='RandomForest',
            opt_num=2
        )
        # Add new points...
    
    # Stage 2: Focused search (Gaussian Process for uncertainty)
    print("Stage 2: Focused Search")
    for i in range(10):
        model = opt.fit(
            data_matrix=data_matrix,
            Measured_response=measured_response,
            virtual_samples=virtual_samples,
            Classifier='GaussianProcess',
            opt_num=1
        )
        # Add new points...
    
    # Stage 3: Local refinement
    print("Stage 3: Local Refinement")
    # Focus on best region...
```

## 模型选择策略

### 集成方法

结合多个模型进行稳健的预测：

```python
def ensemble_optimization(data_matrix, measured_response, virtual_samples):
    """Use ensemble of models for robust optimization."""
    
    models = ['GaussianProcess', 'RandomForest', 'SVR']
    recommendations = []
    
    for model_name in models:
        model = opt.fit(
            data_matrix=data_matrix,
            Measured_response=measured_response,
            virtual_samples=virtual_samples,
            Classifier=model_name,
            opt_num=3
        )
        recommendations.extend(model['recommended_indices'])
    
    # Remove duplicates and select diverse set
    unique_recommendations = list(set(recommendations))
    return unique_recommendations[:5]  # Top 5 diverse recommendations
```

### 自适应模型选择

根据问题特征选择模型：

```python
def adaptive_model_selection(data_matrix, measured_response):
    """Select model based on data characteristics."""
    
    n_samples, n_features = data_matrix.shape
    noise_level = np.std(measured_response) / np.mean(measured_response)
    
    if n_samples < 20:
        return 'GaussianProcess'  # Good for small data
    elif n_features > 10:
        return 'RandomForest'    # Good for high dimensions
    elif noise_level > 0.2:
        return 'SVR'            # Good for noisy data
    else:
        return 'GaussianProcess' # Default choice
```

## 约束处理策略

### 软约束

通过惩罚方法处理约束：

```python
def apply_soft_constraints(virtual_samples, constraints):
    """Apply soft constraints via penalty method."""
    
    penalties = np.zeros(len(virtual_samples))
    
    for i, sample in enumerate(virtual_samples):
        penalty = 0
        
        # Composition constraint
        if np.sum(sample[:3]) > 7.0:
            penalty += 1000 * (np.sum(sample[:3]) - 7.0)
        
        # Ratio constraint
        if sample[1] > 0:  # Avoid division by zero
            ratio = sample[0] / sample[1]
            if ratio < 1.5 or ratio > 4.0:
                penalty += 1000 * abs(ratio - np.clip(ratio, 1.5, 4.0))
        
        penalties[i] = penalty
    
    return penalties

# Use in optimization
penalties = apply_soft_constraints(virtual_samples, constraints)
# Modify acquisition function to include penalties
```

### 硬性约束

过滤虚拟空间以满足约束：

```python
def apply_hard_constraints(virtual_samples):
    """Filter virtual samples to satisfy hard constraints."""
    
    valid_mask = np.ones(len(virtual_samples), dtype=bool)
    
    for i, sample in enumerate(virtual_samples):
        # Check all constraints
        if np.sum(sample[:3]) > 7.0:
            valid_mask[i] = False
        if sample[1] > 0 and not (1.5 <= sample[0]/sample[1] <= 4.0):
            valid_mask[i] = False
        if sample[2] < 0.2:
            valid_mask[i] = False
    
    return virtual_samples[valid_mask]

# Use in optimization
feasible_virtual = apply_hard_constraints(virtual_samples)
model = opt.fit(virtual_samples=feasible_virtual, ...)
```

## 不确定性量化策略

### 引导集成

使用多个模型拟合进行不确定性估计：

```python
def bootstrap_uncertainty(data_matrix, measured_response, virtual_samples, n_bootstrap=10):
    """Estimate uncertainty using bootstrap ensemble."""
    
    predictions = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        n_samples = len(data_matrix)
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        
        bootstrap_data = data_matrix[bootstrap_indices]
        bootstrap_response = measured_response[bootstrap_indices]
        
        # Fit model
        model = opt.fit(
            data_matrix=bootstrap_data,
            Measured_response=bootstrap_response,
            virtual_samples=virtual_samples,
            Classifier='GaussianProcess'
        )
        
        predictions.append(model.virtual_samples_mean)
    
    # Calculate statistics
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred
```

## 性能优化

### 计算效率

更快优化的策略：

```python
# 1. Reduce virtual space size
def smart_virtual_space_reduction(virtual_samples, current_best, reduction_factor=0.5):
    """Intelligently reduce virtual space size."""
    
    # Calculate distances from current best
    distances = np.linalg.norm(virtual_samples - current_best, axis=1)
    
    # Keep closest points and some random distant points
    n_keep = int(len(virtual_samples) * reduction_factor)
    n_close = int(n_keep * 0.7)
    n_random = n_keep - n_close
    
    # Closest points
    close_indices = np.argsort(distances)[:n_close]
    
    # Random distant points
    distant_indices = np.argsort(distances)[n_close:]
    random_distant = np.random.choice(distant_indices, n_random, replace=False)
    
    selected_indices = np.concatenate([close_indices, random_distant])
    return virtual_samples[selected_indices]

# 2. Use faster models for initial screening
def hierarchical_optimization(data_matrix, measured_response, virtual_samples):
    """Use fast models for screening, then GP for refinement."""
    
    # Stage 1: Fast screening with Random Forest
    model_rf = opt.fit(
        data_matrix=data_matrix,
        Measured_response=measured_response,
        virtual_samples=virtual_samples,
        Classifier='RandomForest',
        opt_num=50  # Get many candidates
    )
    
    # Stage 2: Refine with Gaussian Process
    top_candidates = virtual_samples[model_rf['recommended_indices']]
    
    model_gp = opt.fit(
        data_matrix=data_matrix,
        Measured_response=measured_response,
        virtual_samples=top_candidates,
        Classifier='GaussianProcess',
        opt_num=5  # Final selection
    )
    
    return model_gp
```

## 收敛监控

### 跟踪优化进度

```python
def monitor_convergence(optimization_history):
    """Monitor and analyze optimization convergence."""
    
    best_values = []
    improvements = []
    
    for i, result in enumerate(optimization_history):
        if i == 0:
            best_values.append(result)
            improvements.append(0)
        else:
            current_best = max(best_values[-1], result)
            best_values.append(current_best)
            improvements.append(current_best - best_values[-2])
    
    # Check convergence criteria
    recent_improvements = improvements[-5:]  # Last 5 iterations
    avg_improvement = np.mean(recent_improvements)
    
    converged = avg_improvement < 0.01  # Threshold
    
    return {
        'best_values': best_values,
        'improvements': improvements,
        'converged': converged,
        'avg_recent_improvement': avg_improvement
    }
```

## 最佳实践总结

### 数据质量
1. **足够的数据**：每个特征 >10 个样本
2. **代表性抽样**：很好地覆盖设计空间
3. **质量控制**：消除异常值和错误
4. **验证**：使用交叉验证

### 选型
1. **从简单开始**：从高斯过程开始
2. **考虑数据大小**：RF 适用于大数据，GP 适用于小数据
3. **处理噪声**：噪声数据的 SVR
4. **验证性能**：比较多个模型

### 优化策略
1. **多阶段方法**：探索→聚焦→完善
2. **自适应虚拟空间**：根据进度进行调整
3. **约束处理**：选择硬约束与软约束
4. **不确定性量化**：使用 bootstrap 或 GP 不确定性

### 计算效率
1. **智能虚拟空间**：智能缩小尺寸
2. **分层模型**：快速筛选+精准细化
3. **并行实验**：使用批量优化
4. **监控收敛**：收敛时停止

## 常见问题故障排除

### 收敛性差
- 加大早期探索力度
- 检查数据质量和预处理
- 尝试不同的代理模型
- 拓展虚拟空间

### 性能缓慢
- 减小虚拟空间大小
- 使用更简单的模型（随机森林）
- 实施分层优化
- 并行计算

### 违反约束
- 对关键限制使用硬约束
- 对偏好实施软约束
- 验证约束定义
- 检查可行区域大小

## 下一步

- **举例练习**：{doc}`examples/single_objective`
- **学习多目标策略**：{doc}`multibgolearn`
- **探索高级应用**：{doc}`materials_discovery`
- **尝试批量优化**：{doc}`batch_optimization`
