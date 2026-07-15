# BgoFace: Graphical User Interface for Bgolearn

```{note}
[**BgoFace**](https://github.com/Bgolearn/BgoFace) is the graphical user interface (GUI) component of the Bgolearn platform, designed to provide users with an intuitive and efficient way to interact with Bgolearn for material design and optimization.
```

## Overview

BgoFace serves as the user interface for Bgolearn, making the platform more accessible and user-friendly. Through BgoFace, users can easily create optimization tasks, monitor real-time progress, and visualize results using built-in tools. The design focuses on clarity and interactivity, allowing users to focus on optimization tasks without worrying about technical complexities.

```{admonition} Why BgoFace?
:class: tip
BgoFace bridges the gap between experimental and computational domains by:
- **Simplifying complex workflows** - No coding required
- **Providing intuitive controls** - Visual interface for all operations
- **Integrating experimental constraints** - Real-world limitations built-in
- **Enabling seamless access** to active learning algorithms
- **Empowering materials exploration** without deep ML expertise
```

## Key Features

### Dashboard

**Overview Panel**: Provides a snapshot of ongoing optimization tasks, including:
- Task status and progress indicators
- Key performance metrics
- Visual summaries of results
- Real-time updates

### Optimization Management

**Task Monitoring**: 
- Real-time tracking of optimization progress
- Intermediate results display
- Model performance metrics
- Automated notifications and alerts

**Result Analysis**: 
- Comprehensive tools for analyzing optimization results
- Both numerical and graphical interpretations
- Export capabilities for further analysis

### Visualization Tools

**Built-in Plotting Interface**:
- Scatter plots for data exploration
- Line charts for convergence analysis
- Bar graphs for feature importance
- Pareto front visualizations for multi-objective problems
- 3D surface plots for response landscapes

**Data Export Options**:
- PNG/PDF formats for publications
- CSV/Excel for data analysis
- Interactive HTML plots
- High-resolution images for presentations

## Architecture

BgoFace follows a modular architecture that separates the user interface from the computational backend:

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

## Installation and Setup

### Option 1: Download Pre-built Application (Recommended)

For Windows users, the easiest way to get started:

1. **Visit the Releases Page**: [BgoFace Releases](https://github.com/Bgolearn/BgoFace/releases)
2. **Download the Latest Version**: Look for the `.exe` file in the latest release
3. **Run the Application**: No installation required - just double-click to run!

```{admonition} System Requirements
:class: note
- **Operating System**: Windows 10 or later
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 500MB free space
- **Display**: 1024x768 minimum resolution
```

### Option 2: Build from Source

For developers or users who want to customize BgoFace:

#### Prerequisites

```bash
# Install required packages
pip install PyQt5 pyinstaller Bgolearn
```

#### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Bgolearn/BgoFace.git
cd BgoFace

# Install dependencies
pip install -r requirements.txt

# Run from source
python main.py
```

#### Build Executable

```bash
# Create standalone executable
pyinstaller -F -w --add-data "Images;Images" main.py

# The executable will be in the dist/ folder
```

**PyInstaller Options Explained**:
- `-F`: Bundle everything into a single file
- `-w`: Suppress console window (GUI only)
- `--add-data`: Include additional assets like images

## User Interface Guide

### Main Dashboard

The main dashboard provides an overview of your optimization projects:

```python
# Example: Starting a new optimization project
# 1. Click "New Project" button
# 2. Select optimization type (Single/Multi-objective)
# 3. Upload your dataset
# 4. Configure parameters
# 5. Start optimization
```

**Dashboard Components**:
- **Project List**: All your optimization projects
- **Quick Actions**: Start new optimization, import data
- **Recent Results**: Latest optimization outcomes
- **System Status**: Memory usage, computation status

### Data Management

**Data Upload Interface**:
- Drag-and-drop CSV file upload
- Data preview and validation
- Feature selection and target definition
- Data preprocessing options

**Supported Data Formats**:
```csv
# Example dataset format
Feature1,Feature2,Feature3,Target
2.0,1.2,0.5,250
3.5,0.8,0.7,280
1.8,1.5,0.3,240
```

### Optimization Configuration

**Single-Objective Setup**:
1. **Select Target**: Choose the property to optimize
2. **Choose Model**: Gaussian Process, Random Forest, etc.
3. **Set Parameters**: Acquisition function, number of iterations
4. **Define Constraints**: Composition limits, processing ranges
5. **Configure Virtual Space**: Candidate points for evaluation

**Multi-Objective Setup**:
1. **Select Multiple Targets**: Choose 2-6 objectives
2. **Set Optimization Direction**: Maximize/minimize each objective
3. **Choose MOBO Algorithm**: EHVI, PI, or UCB
4. **Configure Pareto Analysis**: Reference points, weights

### Real-Time Monitoring

**Progress Tracking**:
- Optimization iteration counter
- Current best values
- Convergence plots
- Time remaining estimates

**Live Visualization**:
- Acquisition function landscape
- Model predictions vs. actual values
- Pareto front evolution (multi-objective)
- Feature importance updates

### Results Analysis

**Comprehensive Results View**:
- **Summary Statistics**: Best values, improvement metrics
- **Detailed Tables**: All evaluated points with predictions
- **Interactive Plots**: Zoom, pan, and explore results
- **Export Options**: Save plots, data, and reports

## Practical Examples

### Example 1: Alloy Composition Optimization

**Scenario**: Optimize Al-Cu-Mg alloy for maximum strength

**Steps in BgoFace**:

1. **Create New Project**
   - Project Name: "Al-Cu-Mg Strength Optimization"
   - Type: Single-Objective
   - Target: Maximize Strength

2. **Upload Training Data**
   ```csv
   Cu,Mg,Si,Strength
   2.0,1.2,0.5,250
   3.5,0.8,0.7,280
   1.8,1.5,0.3,240
   4.2,0.9,0.8,290
   ```

3. **Configure Optimization**
   - Features: Cu, Mg, Si (composition percentages)
   - Target: Strength (MPa)
   - Model: Gaussian Process
   - Acquisition: Expected Improvement
   - Constraints: Cu+Mg+Si < 7%

4. **Define Virtual Space**
   - Cu range: 1.5-4.5%
   - Mg range: 0.7-1.6%
   - Si range: 0.2-1.0%
   - Generate 1000 candidate compositions

5. **Run Optimization**
   - Monitor progress in real-time
   - View acquisition function landscape
   - Track convergence

6. **Analyze Results**
   - Best composition: Cu=3.8%, Mg=1.0%, Si=0.6%
   - Predicted strength: 295 MPa
   - Export recommendation for experimental validation

### Example 2: Multi-Objective Heat Treatment

**Scenario**: Optimize heat treatment for hardness and toughness

**Steps in BgoFace**:

1. **Create Multi-Objective Project**
   - Objectives: Maximize Hardness, Maximize Toughness
   - Algorithm: EHVI

2. **Upload Process Data**
   ```csv
   Temperature,Time,Cooling_Rate,Hardness,Toughness
   450,2,10,180,45
   500,4,20,220,35
   550,6,15,250,25
   ```

3. **Configure Parameters**
   - Features: Temperature (°C), Time (hours), Cooling Rate (°C/min)
   - Objectives: Hardness (HV), Toughness (J)
   - Constraints: 400°C ≤ Temperature ≤ 600°C

4. **Run Multi-Objective Optimization**
   - Monitor Pareto front evolution
   - View trade-off analysis
   - Track hypervolume improvement

5. **Analyze Pareto Solutions**
   - Explore trade-offs between hardness and toughness
   - Select preferred solution based on requirements
   - Export multiple recommendations

## Advanced Features

### Constraint Handling

**Built-in Constraint Types**:
- **Composition Constraints**: Sum limits, ratio constraints
- **Processing Constraints**: Temperature/pressure ranges
- **Custom Constraints**: User-defined mathematical expressions

**Constraint Definition Interface**:
```python
# Example constraint definitions in BgoFace
constraints = {
    "composition_sum": "Cu + Mg + Si <= 7.0",
    "temperature_range": "400 <= Temperature <= 600",
    "cu_mg_ratio": "1.5 <= Cu/Mg <= 4.0"
}
```

### Batch Optimization

**Parallel Experiment Design**:
- Select multiple points for simultaneous evaluation
- Batch acquisition functions (q-EI, q-UCB)
- Resource allocation optimization
- Experimental planning tools

### Model Comparison

**Automated Model Selection**:
- Compare multiple surrogate models
- Cross-validation performance metrics
- Model uncertainty visualization
- Automatic best model selection

### Export and Reporting

**Comprehensive Export Options**:
- **PDF Reports**: Complete optimization summary
- **Excel Workbooks**: Data tables and charts
- **Python Scripts**: Reproduce analysis programmatically
- **Presentation Slides**: Ready-to-use figures

## Integration with Bgolearn

BgoFace seamlessly integrates with the Bgolearn ecosystem:

### Code Generation

**Automatic Script Generation**:
- BgoFace generates equivalent Python code
- Users can reproduce results programmatically
- Easy transition from GUI to scripting

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

### Data Synchronization

**Seamless Data Flow**:
- Import data from Bgolearn projects
- Export results to Bgolearn format
- Maintain data consistency
- Version control integration

## Best Practices

### Project Organization

1. **Use Descriptive Names**: Clear project and file names
2. **Document Parameters**: Add notes for future reference
3. **Save Intermediate Results**: Regular checkpoints
4. **Organize by Material System**: Group related projects

### Data Quality

1. **Validate Input Data**: Check for outliers and errors
2. **Sufficient Training Data**: >10 samples per feature
3. **Representative Sampling**: Cover the design space well
4. **Quality Control**: Remove experimental errors

### Optimization Strategy

1. **Start Simple**: Begin with single-objective problems
2. **Validate Models**: Use cross-validation
3. **Check Convergence**: Monitor optimization progress
4. **Experimental Validation**: Always test recommendations

## Troubleshooting

### Common Issues

**Application Won't Start**:
- Check system requirements
- Run as administrator (Windows)
- Verify antivirus settings
- Download latest version

**Data Import Problems**:
- Check CSV format and encoding
- Verify column headers
- Remove special characters
- Ensure numeric data types

**Optimization Failures**:
- Reduce virtual space size
- Check constraint definitions
- Verify data quality
- Try different surrogate models

**Performance Issues**:
- Close other applications
- Reduce dataset size
- Use simpler models
- Increase available memory

### Getting Help

**Support Resources**:
- **User Manual**: Detailed PDF guide included
- **Video Tutorials**: [BiliBili Channel](https://www.bilibili.com/video/BV1LTtLeaEZp/)
- **GitHub Issues**: [Report bugs and request features](https://github.com/Bgolearn/BgoFace/issues)
- **Email Support**: Contact the development team

**Community**:
- **Discussion Forum**: Share experiences and tips
- **Example Projects**: Download sample datasets
- **Best Practices**: Learn from other users

## Future Development

### Planned Features

**Version 2.0 Roadmap**:
- **Cloud Integration**: Remote computation support
- **Collaborative Features**: Multi-user projects
- **Advanced Visualization**: 3D interactive plots
- **Machine Learning**: Automated hyperparameter tuning
- **Integration**: Connect with experimental equipment

**Community Contributions**:
- Plugin system for custom algorithms
- Template library for common materials
- Shared project repository
- Educational resources

## Citation and License

### Academic Use

If you use BgoFace in your research, please cite:

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

### License

**Academic and Research Use Only**
- ✅ Academic research and education
- ✅ Non-commercial scientific studies
- ✅ Open-source contributions
- ❌ Commercial applications
- ❌ Proprietary software integration
- ❌ Revenue-generating activities

**Copyright**: © 2024 Bgolearn Development Team. All rights reserved.


**Development Team**:
- **Lead Developer**: Bin Cao (Hong Kong University of Science and Technology, Guangzhou)
- **UI Development**: Siyuan Liu (Tongji University), Bin Cao (HKUST, Guangzhou)
- **Guidance**: Prof. Tong-Yi Zhang (Shanghai University), Prof. Lingyan Feng (Shanghai University)

## Next Steps

Ready to start using BgoFace? Here's what to do next:

1. **Download BgoFace**: Get the latest version from [GitHub Releases](https://github.com/Bgolearn/BgoFace/releases)
2. **Watch Tutorial**: Follow the [video guide](https://www.bilibili.com/video/BV1LTtLeaEZp/)
3. **Try Examples**: Start with the included sample projects
4. **Join Community**: Connect with other users and developers
5. **Explore Advanced Features**: Learn about multi-objective optimization and constraints

```{seealso}
Related Documentation:
- {doc}`getting_started` - Install and setup Bgolearn
- {doc}`multibgolearn` - Multi-objective optimization
- {doc}`materials_discovery` - Materials discovery examples
- {doc}`visualization` - Advanced plotting techniques
```
