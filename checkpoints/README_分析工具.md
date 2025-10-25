# 检查点分析工具使用说明

## 📋 概述

`analyze_checkpoint.py` 是一个全面的DiffSBDD模型检查点分析工具，能够深入解析模型结构、参数配置和训练状态。

## 🎯 功能特性

### 1. 文件信息
- 文件大小和路径
- 修改时间
- 基本元数据

### 2. 模型架构分析
- **总参数统计**: 精确计算可训练参数数量
- **模块分解**: 按模块统计参数分布
- **EGNN层结构**: 详细分析每一层的参数和子模块
- **参数详情**: 每个参数的形状、数量和数据类型

### 3. 超参数配置
- **训练参数**: batch_size, 学习率, 数据增强等
- **EGNN参数**: 层数, 隐藏维度, 注意力机制等
- **扩散参数**: 扩散步数, 噪声调度, 损失类型等
- **模型配置**: 模型模式, 口袋表示方式, 数据集等

### 4. 训练状态
- **优化器状态**: 学习率, betas, weight_decay等
- **学习率调度器**: 调度策略和当前状态
- **回调函数**: 最佳模型路径和得分
- **训练进度**: 当前epoch和全局步数

### 5. 输出格式
- **JSON报告**: 结构化数据，便于程序解析
- **Markdown报告**: 人类可读的详细分析文档

## 📦 依赖要求

- Python 3.7+
- PyTorch
- NumPy

## 🚀 快速开始

### 基本用法

```bash
# 分析检查点
python analyze_checkpoint.py crossdocked_fullatom_cond.ckpt

# 输出文件:
#   - crossdocked_fullatom_cond_analysis.json
#   - crossdocked_fullatom_cond_analysis.md
```

### 自定义输出文件名

```bash
python analyze_checkpoint.py crossdocked_fullatom_cond.ckpt -o my_analysis

# 输出文件:
#   - my_analysis.json
#   - my_analysis.md
```

### 查看帮助

```bash
python analyze_checkpoint.py -h
```

## 📊 分析报告示例

### 执行摘要

从 `crossdocked_fullatom_cond.ckpt` 的分析结果：

| 指标 | 值 |
|------|------|
| **总参数数** | 1,006,560 (约100万) |
| **参数大小** | 3.84 MB |
| **EGNN层数** | 5层 |
| **隐藏层维度** | 128 |
| **注意力机制** | 启用 |
| **扩散步数** | 500 |
| **训练轮数** | 999 |
| **全局步数** | 1,562,000 |

### EGNN 层结构

每层参数分布均匀：

| 层编号 | 参数数量 | 子模块数 |
|--------|----------|----------|
| Layer 0 | 198,785 | 20 |
| Layer 1 | 198,785 | 20 |
| Layer 2 | 198,785 | 20 |
| Layer 3 | 198,785 | 20 |
| Layer 4 | 198,785 | 20 |

**总计**: 5层 × 19.9万参数/层 ≈ 99.4万参数

## 💡 使用场景

### 1. 迭代学习前的分析

在使用 `iterative_learning` 进行迭代训练前，先分析检查点：

```bash
cd checkpoints
python analyze_checkpoint.py crossdocked_fullatom_cond.ckpt
```

**分析报告帮助你：**
- 确定合适的冻结层数（模型共5层，推荐冻结前3层）
- 了解模型规模，评估训练时间
- 验证检查点完整性

### 2. 对比不同检查点

分析多个检查点以对比差异：

```bash
python analyze_checkpoint.py checkpoint_v1.ckpt -o v1_analysis
python analyze_checkpoint.py checkpoint_v2.ckpt -o v2_analysis

# 对比两个JSON文件
diff v1_analysis.json v2_analysis.json
```

### 3. 调试训练问题

检查点损坏或训练异常时，使用分析工具诊断：

```bash
python analyze_checkpoint.py problematic_checkpoint.ckpt

# 检查输出中的:
# - 参数数量是否正确
# - 优化器状态是否完整
# - 训练进度是否异常
```

### 4. 模型结构研究

深入理解DiffSBDD模型架构：

```bash
python analyze_checkpoint.py crossdocked_fullatom_cond.ckpt
# 查看详细参数列表，了解每个模块的作用
```

## 📖 输出文件说明

### JSON报告 (*.json)

结构化数据，便于程序解析：

```json
{
  "file_info": {...},
  "structure": {...},
  "hyperparameters": {...},
  "architecture": {
    "总参数数": "1,006,560",
    "EGNN层分析": {
      "num_layers": 5,
      "layers": [...]
    }
  },
  "optimizer": {...},
  "statistics": {...}
}
```

**用途**：
- 自动化分析脚本
- 批量对比检查点
- 生成统计图表

### Markdown报告 (*.md)

人类可读的详细文档，包含：

1. **执行摘要**: 关键指标一览
2. **文件信息**: 基本元数据
3. **检查点结构**: 顶层键和训练状态
4. **超参数配置**: 详细的配置表格
5. **模型架构分析**: 参数分布和层结构
6. **优化器状态**: 学习率和优化器配置
7. **详细参数列表**: 完整的参数清单（可折叠）
8. **结论与建议**: 基于分析的实用建议

**用途**：
- 快速了解模型概况
- 文档归档
- 团队分享

## 🔧 高级用法

### 在代码中使用

```python
from analyze_checkpoint import CheckpointAnalyzer

# 创建分析器
analyzer = CheckpointAnalyzer('crossdocked_fullatom_cond.ckpt')

# 执行分析
results = analyzer.analyze_all()

# 访问分析结果
print(f"总参数数: {results['statistics']['总参数数']}")
print(f"EGNN层数: {results['statistics']['EGNN层数']}")

# 保存报告
analyzer.save_json_report('output.json')
analyzer.generate_markdown_report('output.md')
```

### 自定义分析

修改 `analyze_checkpoint.py` 添加自定义分析：

```python
def analyze_custom_metric(self):
    """自定义分析函数"""
    # 你的分析逻辑
    custom_results = {...}
    self.analysis_results['custom'] = custom_results
    return custom_results
```

## ⚠️ 注意事项

### 1. 兼容性

- **PyTorch版本**: 支持PyTorch 1.x 和 2.x
- **检查点格式**: 仅支持PyTorch Lightning格式的检查点
- **Python版本**: 需要Python 3.7+

### 2. 内存使用

- 分析大型检查点（>100M）时可能需要较多内存
- 工具会在CPU上加载检查点，避免占用GPU

### 3. 安全性

- 脚本使用 `weights_only=False` 加载检查点
- 仅分析来自可信来源的检查点文件

## 🐛 故障排查

### 问题1: ModuleNotFoundError

```bash
ModuleNotFoundError: No module named 'torch'
```

**解决方案**:
```bash
# 激活DiffSBDD环境
conda activate diffsbdd

# 或安装PyTorch
pip install torch
```

### 问题2: 无法加载检查点

```bash
RuntimeError: 加载检查点失败
```

**解决方案**:
- 检查文件路径是否正确
- 确认检查点文件未损坏
- 确保有足够的磁盘空间和内存

### 问题3: Unicode编码错误

```bash
UnicodeEncodeError: 'gbk' codec can't encode...
```

**解决方案**:
- 这是Windows中文环境的已知问题
- 脚本已修复，使用ASCII字符代替emoji
- 如仍有问题，请使用英文路径

## 📚 参考资料

- [DiffSBDD项目主页](https://github.com/arneschneuing/DiffSBDD)
- [PyTorch Lightning文档](https://pytorch-lightning.readthedocs.io/)
- [iterative_learning使用说明](../iterative_learning/README.md)

## 🤝 贡献

欢迎提出改进建议！可以：

1. 添加新的分析指标
2. 改进可视化效果
3. 支持更多检查点格式
4. 优化性能和内存使用

## 📄 许可证

本工具继承DiffSBDD项目的许可证。

---

**版本**: 1.0  
**更新日期**: 2025-10-25  
**作者**: DiffSBDD迭代学习项目组

**祝你分析愉快！** 🚀

