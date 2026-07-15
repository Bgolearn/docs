# Bgolearn Documentation Usage Guide

## 📁 文件夹结构

所有文档文件现在都在 `bgolearnbook/` 文件夹中：

```
bgolearnbook/
├── _config.yml                    # Jupyter Book 配置
├── _toc.yml                      # 目录结构
├── index.md                      # 主页
├── getting_started.md            # 入门指南
├── basic_concepts.md             # 基础概念
├── first_optimization.md         # 第一次优化教程
├── api_reference.md             # API 参考
├── acquisition_functions.md      # 获取函数指南
├── batch_optimization.md        # 批量优化
├── visualization.md             # 可视化
├── materials_discovery.md       # 材料发现
├── examples/                    # 示例
│   ├── basic/
│   │   └── simple_1d.ipynb     # 1D优化示例
│   └── materials/
│       └── alloy_optimization.ipynb # 合金优化示例
├── images/                      # 图片资源
├── requirements.txt             # 依赖项
├── references.bib              # 参考文献
├── Makefile                    # 构建脚本
├── build_docs.sh              # 构建脚本
├── test_build.py              # 测试脚本
└── README.md                   # 说明文档
```

## 🚀 快速开始

### 方法1: 使用构建脚本（推荐）

```bash
cd bgolearnbook
./build_docs.sh
```

这个脚本会：
1. 安装依赖项
2. 清理旧的构建文件
3. 构建 Jupyter Book
4. 可选择启动本地服务器

### 方法2: 使用 Makefile

```bash
cd bgolearnbook

# 安装依赖
make install

# 构建文档
make build

# 启动本地服务器
make serve
```

### 方法3: 手动构建

```bash
cd bgolearnbook

# 安装依赖
pip install -r requirements.txt

# 构建文档
jupyter-book build .

# 启动本地服务器
cd _build/html && python -m http.server 8000
```

## 🧪 测试文档结构

在构建之前，可以运行测试脚本检查文档结构：

```bash
cd bgolearnbook
python test_build.py
```

这会检查：
- 配置文件是否有效
- 必需文件是否存在
- 示例文件是否完整
- 依赖项是否安装

## 📖 查看文档

构建完成后，在浏览器中打开：
- `http://localhost:8000` （如果使用本地服务器）
- 或直接打开 `bgolearnbook/_build/html/index.html`

## 🛠️ 可用命令

### Makefile 命令

```bash
make help          # 显示所有可用命令
make install       # 安装依赖项
make build         # 构建文档
make serve         # 构建并启动服务器
make clean         # 清理构建文件
make check         # 检查链接
make pdf           # 构建PDF版本
make publish       # 发布到GitHub Pages
```

### 构建脚本

```bash
./build_docs.sh    # 交互式构建和服务
```

### 测试脚本

```bash
python test_build.py    # 验证文档结构
```

## 📝 添加新内容

### 添加新页面

1. 在 `bgolearnbook/` 中创建新的 `.md` 文件
2. 在 `_toc.yml` 中添加条目
3. 重新构建文档

### 添加新的 Jupyter Notebook

1. 在 `examples/` 相应子目录中创建 `.ipynb` 文件
2. 在 `_toc.yml` 中添加条目
3. 重新构建文档

### 修改配置

- 编辑 `_config.yml` 修改网站设置
- 编辑 `_toc.yml` 修改目录结构
- 编辑 `requirements.txt` 添加新依赖

## 🎨 自定义

### 主题和样式

在 `_config.yml` 中修改：
```yaml
html:
  theme: sphinx_book_theme
  # 其他主题选项
```

### 扩展功能

在 `_config.yml` 中添加 Sphinx 扩展：
```yaml
sphinx:
  extra_extensions:
    - sphinx_copybutton
    - sphinx_design
```

## 🔧 故障排除

### 常见问题

1. **构建失败**
   ```bash
   make clean && make build
   ```

2. **依赖项缺失**
   ```bash
   pip install -r requirements.txt
   ```

3. **Notebook 执行错误**
   ```bash
   jupyter-book build . --execute-timeout 300
   ```

4. **端口被占用**
   ```bash
   # 使用不同端口
   python -m http.server 8080
   ```

### 获取帮助

- 检查 Jupyter Book 文档: https://jupyterbook.org/
- 运行 `python test_build.py` 诊断问题
- 查看构建日志中的错误信息

## 📊 文档特性

### 已实现的功能

✅ **基础功能**
- Jupyter Book 配置
- 响应式设计
- 搜索功能
- 代码高亮

✅ **内容**
- 入门教程
- API 文档
- 交互式示例
- 材料科学案例

✅ **高级功能**
- MyST 语法支持
- 交互式代码执行
- 数学公式渲染
- 参考文献管理

### 可扩展功能

🔄 **可添加的功能**
- 更多示例 Notebook
- 视频教程嵌入
- 多语言支持
- 评论系统
- 分析统计

## 🎯 下一步

1. **测试文档**: `python test_build.py`
2. **构建文档**: `./build_docs.sh`
3. **查看结果**: 在浏览器中打开
4. **添加内容**: 根据需要扩展文档
5. **发布**: 使用 `make publish` 发布到 GitHub Pages

现在您有了一个完整的 Jupyter Book 文档系统，可以为 Bgolearn 提供专业的在线文档！
