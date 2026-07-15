# BgoFace：Bgolearn 的图形用户界面

```{note}
本页是 Bgolearn 手册的中文版本。
```

```{note}
[**BgoFace**](https://github.com/Bgolearn/BgoFace) 是 Bgolearn 平台的图形用户界面 (GUI) 组件，旨在为用户提供直观、高效的方式与 Bgolearn 进行交互，以进行材料设计和优化。
```

## 概述

BgoFace 作为 Bgolearn 的用户界面，使平台更易于访问和用户友好。通过BgoFace，用户可以轻松创建优化任务、监控实时进度并使用内置工具可视化结果。该设计注重清晰度和交互性，让用户能够专注于优化任务，而不必担心技术复杂性。

```{admonition} 为什么使用 BgoFace？
:class: tip
BgoFace 通过以下方式弥合了实验和计算领域之间的差距：
- **简化复杂的工作流程** - 无需编码
- **提供直观的控制** - 所有操作的可视化界面
- **整合实验约束** - 内置现实世界的限制
- **实现主动学习算法的无缝访问**
- **赋能材料探索**，无需深厚的机器学习专业知识
```

## 主要特点

### 仪表板

**概述面板**：提供正在进行的优化任务的快照，包括：
- 任务状态和进度指示器
- 关键绩效指标
- 结果的可视化总结
- 实时更新

### 优化管理

**任务监控**： 
- 实时跟踪优化进度
- 中间结果显示
- 模型性能指标
- 自动通知和警报

**结果分析**： 
- 用于分析优化结果的综合工具
- 数字和图形解释
- 导出功能以供进一步分析

### 可视化工具

**内置绘图界面**：
- 用于数据探索的散点图
- 用于收敛分析的折线图
- 特征重要性的条形图
- 多目标问题的帕累托前沿可视化
- 响应景观的 3D 曲面图

**数据导出选项**：
- 用于出版物的 PNG/PDF 格式
- 用于数据分析的 CSV/Excel
- 交互式 HTML 绘图
- 用于演示的高分辨率图像

## 建筑学

BgoFace 遵循模块化架构，将用户界面与计算后端分开：

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│   BgoFace GUI   │───▶│  Bgolearn Core  │
│                 │    │                 │    │                 │
│ • Parameters    │    │ • Task Manager  │    │ • Optimization  │
│ • Data Upload   │    │ • Visualizer    │    │ • ML Models     │
│ • Constraints   │    │ • Result Viewer │    │ • Algorithms    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   File System   │    │   Computation   │
                       │                 │    │                 │
                       │ • Data Storage  │    │ • Model Training│
                       │ • Results Cache │    │ • Predictions   │
                       │ • Export Files  │    │ • Optimization  │
                       └─────────────────┘    └─────────────────┘
```

## 安装和设置

### 选项 1：下载预构建的应用程序（推荐）

对于 Windows 用户，最简单的入门方法：

1. **访问版本页面**：[BgoFace 版本](https://github.com/Bgolearn/BgoFace/releases)
2. **Download the Latest Version**: Look for the `.exe` file in the latest release
3. **运行应用程序**：无需安装 - 只需双击即可运行！

```{admonition} 系统要求
:class: note
- **操作系统**：Windows 10 或更高版本
- **内存**：最低 4GB RAM，建议 8GB
- **存储**：500MB 可用空间
- **显示**：最小分辨率 1024x768
```

### 选项 2：从源代码构建

对于想要定制BgoFace的开发者或用户：

#### 先决条件

```bash
# Install required packages
pip install PyQt5 pyinstaller Bgolearn
```

#### 克隆和设置

```bash
# Clone the repository
git clone https://github.com/Bgolearn/BgoFace.git
cd BgoFace

# Install dependencies
pip install -r requirements.txt

# Run from source
python main.py
```

#### 构建可执行文件

```bash
# Create standalone executable
pyinstaller -F -w --add-data "Images;Images" main.py

# The executable will be in the dist/ folder
```

**PyInstaller 选项解释**：
- `-F`：将所有内容捆绑到一个文件中
- `-w`：抑制控制台窗口（仅限 GUI）
- `--add-data`：包括图像等附加资源

## 用户界面指南

### 主仪表板

主仪表板提供了优化项目的概述：

```python
# Example: Starting a new optimization project
# 1. Click "New Project" button
# 2. Select optimization type (Single/Multi-objective)
# 3. Upload your dataset
# 4. Configure parameters
# 5. Start optimization
```

**仪表板组件**：
- **项目列表**：您所有的优化项目
- **快速操作**：开始新的优化，导入数据
- **最新结果**：最新优化结果
- **系统状态**：内存使用情况、计算状态

### 数据管理

**数据上传接口**：
- 拖放 CSV 文件上传
- 数据预览和验证
- 特征选择和目标定义
- 数据预处理选项

**支持的数据格式**：
```csv
# Example dataset format
Feature1,Feature2,Feature3,Target
2.0,1.2,0.5,250
3.5,0.8,0.7,280
1.8,1.5,0.3,240
```

### 优化配置

**单目标设置**：
1. **选择目标**：选择要优化的属性
2. **选择模型**：高斯过程、随机森林等。
3. **设置参数**：采集函数、迭代次数
4. **定义约束**：成分限制、加工范围
5. **配置虚拟空间**：评估候选点

**多目标设置**：
1. **选择多个目标**：选择2-6个目标
2. **设置优化方向**：最大化/最小化每个目标
3. **选择 MOBO 算法**：EHVI、PI 或 UCB
4. **配置帕累托分析**：参考点、权重

### 实时监控

**进度跟踪**：
- 优化迭代计数器
- 当前最佳值
- 收敛图
- 剩余时间估计

**实时可视化**：
- 采集功能景观
- 模型预测与实际值
- 帕累托前沿演化（多目标）
- 功能重要性更新

### 结果分析

**综合结果视图**：
- **统计摘要**：最佳值、改进指标
- **详细表格**：所有带有预测的评估点
- **交互式绘图**：缩放、平移和探索结果
- **导出选项**：保存绘图、数据和报告

## 实际例子

### 示例1：合金成分优化

**场景**：优化 Al-Cu-Mg 合金以获得最大强度

**BgoFace 中的步骤**：

1. **创建新项目**
   - 项目名称：“Al-Cu-Mg强度优化”
   - 类型：单目标
   - 目标：力量最大化

2. **上传训练数据**
   ```csv
   Cu,Mg,Si,Strength
   2.0,1.2,0.5,250
   3.5,0.8,0.7,280
   1.8,1.5,0.3,240
   4.2,0.9,0.8,290
   ```

3. **配置优化**
   - 特性：Cu、Mg、Si（成分百分比）
   - 目标：强度（MPa）
   - 模型：高斯过程
   - 收购：预期改善
   - 限制条件：Cu+Mg+Si < 7%

4. **定义虚拟空间**
   - 铜范围：1.5-4.5%
   - 镁范围：0.7-1.6%
   - Si范围：0.2-1.0%
   - 生成 1000 个候选作品

5. **运行优化**
   - 实时监控进度
   - 查看采集功能景观
   - 轨道收敛

6. **分析结果**
   - 最佳成分：Cu=3.8%、Mg=1.0%、Si=0.6%
   - 预测强度：295 MPa
   - 用于实验验证的导出建议

### 示例 2：多目标热处理

**场景**：优化热处理以提高硬度和韧性

**BgoFace 中的步骤**：

1. **创建多目标项目**
   - 目标：最大化硬度、最大化韧性
   - 算法：EHVI

2. **上传过程数据**
   ```csv
   Temperature,Time,Cooling_Rate,Hardness,Toughness
   450,2,10,180,45
   500,4,20,220,35
   550,6,15,250,25
   ```

3. **配置参数**
   - 特征：温度（°C）、时间（小时）、冷却速率（°C/分钟）
   - 目标：硬度 (HV)、韧性 (J)
   - 限制条件：400°C ≤ 温度 ≤ 600°C

4. **运行多目标优化**
   - 监控帕累托前沿演化
   - 查看权衡分析
   - 跟踪超容量改进

5. **分析帕累托解**
   - 探索硬度和韧性之间的权衡
   - 根据需求选择首选解决方案
   - 导出多个推荐

## 高级功能

### 约束处理

**内置约束类型**：
- **成分限制**：总和限制、比率限制
- **加工限制**：温度/压力范围
- **自定义约束**：用户定义的数学表达式

**约束定义接口**：
```python
# Example constraint definitions in BgoFace
constraints = {
    "composition_sum": "Cu + Mg + Si <= 7.0",
    "temperature_range": "400 <= Temperature <= 600",
    "cu_mg_ratio": "1.5 <= Cu/Mg <= 4.0"
}
```

### 批量优化

**并行实验设计**：
- 选择多个点同时评估
- 批量采集功能（q-EI、q-UCB）
- 资源配置优化
- 实验规划工具

### 型号对比

**自动模型选择**：
- 比较多个替代模型
- 交叉验证性能指标
- 模型不确定性可视化
- 自动最佳模型选择

### 导出和报告

**全面的导出选项**：
- **PDF 报告**：完整的优化摘要
- **Excel 工作簿**：数据表和图表
- **Python 脚本**：以编程方式重现分析
- **演示幻灯片**：即用型图形

## 与 Bgolearn 集成

BgoFace与Bgolearn生态系统无缝集成：

### 代码生成

**自动脚本生成**：
- BgoFace 生成等效的 Python 代码
- 用户可以通过编程方式重现结果
- 从 GUI 轻松过渡到脚本编写

```python
# Example generated code from BgoFace
from Bgolearn import BGOsampling

# Initialize optimizer
opt = BGOsampling.Bgolearn()

# Configuration from BgoFace
model = opt.fit(
    data_matrix=data_matrix,
    Measured_response=measured_response,
    virtual_samples=virtual_samples,
    Mission='Regression',
    Classifier='GaussianProcess',
    opt_num=1,
    min_search=False,
    CV_test=10,  # 10-fold cross-validation
    Normalize=True
)
```

### 数据同步

**无缝数据流**：
- 从 Bgolearn 项目导入数据
- 将结果导出为 Bgolearn 格式
- 保持数据一致性
- 版本控制集成

## 最佳实践

### 项目组织

1. **使用描述性名称**：清除项目和文件名
2. **文档参数**：添加注释以供将来参考
3. **保存中间结果**：定期检查点
4. **按材料系统组织**：组相关项目

### 数据质量

1. **验证输入数据**：检查异常值和错误
2. **足够的训练数据**：每个特征 >10 个样本
3. **代表性抽样**：很好地覆盖设计空间
4. **质量控制**：消除实验错误

### 优化策略

1. **从简单开始**：从单目标问题开始
2. **验证模型**：使用交叉验证
3. **检查收敛**：监控优化进度
4. **实验验证**：始终测试建议

## 故障排除

### 常见问题

**应用程序无法启动**：
- 检查系统要求
- 以管理员身份运行 (Windows)
- 验证防病毒设置
- 下载最新版本

**数据导入问题**：
- 检查 CSV 格式和编码
- 验证列标题
- 删除特殊字符
- 确保数字数据类型

**优化失败**：
- 减小虚拟空间大小
- 检查约束定义
- 验证数据质量
- 尝试不同的代理模型

**性能问题**：
- 关闭其他应用程序
- 减小数据集大小
- 使用更简单的模型
- 增加可用内存

### 寻求帮助

**支持资源**：
- **用户手册**：包含详细的 PDF 指南
- **视频教程**：【哔哩哔哩频道】(https://www.bilibili.com/video/BV1LTtLeaEZp/)
- **GitHub 问题**：[报告错误并请求功能](https://github.com/Bgolearn/BgoFace/issues)
- **电子邮件支持**：联系开发团队

**社区**：
- **讨论论坛**：分享经验和技巧
- **示例项目**：下载示例数据集
- **最佳实践**：向其他用户学习

## 未来发展

### 计划的功能

**2.0 版路线图**：
- **云集成**：远程计算支持
- **协作功能**：多用户项目
- **高级可视化**：3D 交互式绘图
- **机器学习**：自动超参数调整
- **集成**：与实验设备连接

**社区贡献**：
- 用于自定义算法的插件系统
- 常用材质模板库
- 共享项目存储库
- 教育资源

## 引用和许可

### 学术用途

如果您在研究中使用 BgoFace，请引用：

```bibtex
@article{li2025optimize,
  title={Optimize the quantum yield of G-quartet-based circularly polarized luminescence materials via active learning strategy-BgoFace},
  author={Li, Tianliang and Chen, Lifei and Cao, Bin and Liu, Siyuan and Lin, Lixing and Li, Zeyu and Chen, Yingying and Li, Zhenzhen and Zhang, Tong-yi and Feng, Lingyan},
  journal={Materials Genome Engineering Advances},
  volume={3},
  number={3},
  pages={e70031},
  year={2025},
  publisher={Wiley Online Library}
}
```

### 执照

**仅供学术和研究使用**
- ✅ 学术研究和教育
- ✅ 非商业性科学研究
- ✅ 开源贡献
- ❌商业应用
- ❌ 专有软件集成
- ❌ 创收活动

**版权**：© 2024 Bgolearn 开发团队。版权所有。


**开发团队**：
- **首席开发人员**：曹斌（香港科技大学，广州）
- **UI开发**：Siyuan Liu（同济大学）、Bin Cao（香港科技大学，广州）
- **指导**：张同义教授（上海大学）、冯凌燕教授（上海大学）

## 下一步

准备好开始使用 BgoFace 了吗？接下来要做的事情如下：

1. **下载BgoFace**：从[GitHub发布](https://github.com/Bgolearn/BgoFace/releases)获取最新版本
2. **观看教程**：按照[视频指南](https://www.bilibili.com/video/BV1LTtLeaEZp/)
3. **尝试示例**：从包含的示例项目开始
4. **加入社区**：与其他用户和开发人员联系
5. **探索高级功能**：了解多目标优化和约束

```{seealso}
相关文档：
- {doc}`getting_started` - 安装和设置 Bgolearn
- {doc}`multibgolearn` - 多目标优化
- {doc}`materials_discovery` - 材料发现示例
- {doc}`visualization` - 高级绘图技术
```
