# 文件索引 - 完整清单

## 📋 目录

1. [核心程序](#核心程序) - Python源代码
2. [脚本工具](#脚本工具) - Shell脚本
3. [文档](#文档) - 说明文档
4. [工作目录](#工作目录) - 数据和输出

---

## 核心程序

### 1. iterative_generation.py
**用途**: 主程序，协调整个迭代学习流程

**核心功能**:
- IterativeLearning类：主控制器
- generate_molecules(): 批量生成分子
- select_best_molecules(): 评估并选择最优分子（集成不确定性选择）⭐
- train_on_selected_molecules(): 训练模型
- run_iteration(): 执行单次迭代
- run(): 运行完整的30次迭代

**依赖**:
- molecule_evaluator.py
- uncertainty_selector.py ⭐ NEW
- prepare_training_data.py
- train_frozen.py
- DiffSBDD核心模块

**输入**:
- 预训练模型检查点
- 蛋白PDB文件
- 配置参数

**输出**:
- 生成的分子（SDF文件）
- 训练的模型
- 评分记录
- 最终报告

**何时使用**: 
- 作为主入口运行整个流程
- 或被run_example.sh调用

---

### 2. molecule_evaluator.py
**用途**: 分子评估器，计算多维度评分

**核心功能**:
- MoleculeEvaluator类：评估器
- calculate_qed(): 计算类药性
- calculate_sa(): 计算合成可及性
- calculate_logp(): 计算亲脂性
- calculate_lipinski(): 计算Lipinski规则
- calculate_docking_score(): 对接打分（可选）
- calculate_综合得分(): 加权综合评分

**评分体系**:
```
综合得分 = 0.25×QED + 0.25×SA + 0.15×LogP + 0.15×Lipinski + 0.20×Docking
```

**输入**:
- RDKit分子对象列表
- 蛋白PDB文件（用于对接）

**输出**:
- Pandas DataFrame（包含所有评分）

**何时修改**:
- 调整评分权重
- 添加新的评估指标
- 修改归一化方法

---

### 3. uncertainty_selector.py ⭐ NEW
**用途**: 基于不确定性的智能选择策略

**核心功能**:
- UncertaintyBasedSelector类：不确定性选择器
- compute_alpha(): 计算探索权重（线性衰减）
- compute_fingerprints(): 计算Morgan指纹
- compute_uncertainty_scores(): 计算不确定性得分
- select_molecules(): 综合质量和不确定性选择分子
- _update_known_space(): 更新已知化学空间

**核心算法**:
```
综合得分 = (1-α) × 质量得分 + α × 不确定性得分
不确定性 = 1 - max_similarity(分子, 已知空间)
α(t) = α_start - (α_start - α_end) × (t-1)/(T-1)
```

**工作原理**:
- 前期（迭代1-10）：α≈0.5，大胆探索未知化学空间
- 中期（迭代11-20）：α≈0.25，平衡探索与利用
- 后期（迭代21-30）：α≈0.03，聚焦精细优化

**输入**:
- RDKit分子对象列表
- 分子质量评分DataFrame
- 已知化学空间（历史选择记录）

**输出**:
- 选中的分子列表
- 详细评分DataFrame（包含不确定性指标）
- 统计分析JSON文件

**何时修改**:
- 调整衰减策略（线性→指数/余弦）
- 修改相似度计算方法
- 添加多目标优化

---

### 4. prepare_training_data.py
**用途**: 训练数据准备，将分子转换为NPZ格式

**核心功能**:
- DataPreparer类：数据转换器
- encode_ligand(): 编码配体
- encode_pocket(): 编码口袋
- center_complex(): 居中复合物
- prepare_iterative_training_data(): 主函数

**数据转换流程**:
```
RDKit分子 → 坐标+类型 → One-hot编码 → 与口袋组合 → NPZ文件
```

**NPZ文件格式**:
- lig_coords: (N, 3) 配体坐标
- lig_one_hot: (N, atom_types) 配体类型
- lig_mask: (N,) 批次掩码
- pocket_coords: (M, 3) 口袋坐标
- pocket_one_hot: (M, residue_types) 口袋类型
- pocket_mask: (M,) 批次掩码

**何时使用**:
- 被iterative_generation.py自动调用
- 或单独使用准备自定义数据集

---

### 5. train_frozen.py
**用途**: 冻结层训练，固定底层只训练上层

**核心功能**:
- freeze_model_layers(): 冻结指定层
- train_with_frozen_layers(): 执行训练

**冻结策略**:
```
EGNN (5层):
├── Layer 0-2: 冻结（保持预训练知识）
└── Layer 3-4: 训练（适应特定蛋白）
```

**训练配置**:
- 优化器: AdamW
- 学习率: 1e-4（可调）
- 早停: patience=10
- 检查点: 保存最佳模型

**何时修改**:
- 调整冻结层数
- 修改学习率
- 添加学习率调度器

---

## 脚本工具

### 5. run_example.sh
**用途**: 运行脚本，一键启动迭代学习

**功能**:
- 配置所有参数
- 检查文件存在性
- 调用iterative_generation.py

**关键参数**:
```bash
CHECKPOINT    # 模型路径
PDBFILE       # 蛋白文件
OUTPUT_DIR    # 输出目录
REF_LIGAND    # 口袋定义
N_ITERATIONS  # 迭代次数
TRAIN_EPOCHS  # 训练轮数
```

**使用方法**:
```bash
# 1. 编辑参数
nano run_example.sh

# 2. 运行
./run_example.sh
```

---

### 6. setup_environment.sh
**用途**: 环境设置，自动配置运行环境

**功能**:
- 创建conda环境
- 下载预训练模型
- 创建目录结构
- 安装smina（可选）

**使用场景**:
- 首次使用时运行
- 重置环境

**使用方法**:
```bash
./setup_environment.sh
```

---

### 7. check_setup.py
**用途**: 环境检查，验证配置是否正确

**检查项**:
- ✓ Python版本
- ✓ 必需包（PyTorch, RDKit等）
- ✓ CUDA可用性
- ✓ 核心文件存在
- ✓ DiffSBDD模块
- ✓ 预训练模型
- ✓ Smina（可选）

**使用方法**:
```bash
conda activate diffsbdd
python check_setup.py
```

**输出示例**:
```
====================================================================
检查总结
====================================================================
Python版本            ✓ 通过
Python包              ✓ 通过
CUDA                  ✓ 通过
...
====================================================================
```

---

## 文档

### 8. README.md
**语言**: 英文  
**内容**: 完整详细的技术文档

**章节**:
- 概述和工作流程
- 安装依赖
- 快速开始
- 参数说明
- 评分系统
- 输出文件说明
- 自定义配置
- 性能优化
- 故障排除
- 进阶使用

**适合**: 深入了解技术细节

---

### 9. 使用说明.txt
**语言**: 中文  
**内容**: 简明快速指南

**章节**:
- 项目简介
- 快速开始（4步）
- 迭代流程
- 评分系统
- 输出结果
- 常见问题
- 文件说明

**适合**: 快速上手

---

### 10. 快速开始指南.md
**语言**: 中文  
**内容**: 详细的逐步教程

**章节**:
- 30秒快速开始
- 详细步骤（6步）
- 文件结构说明
- 常见问题Q&A
- 高级用法
- 性能基准
- 下一步建议

**适合**: 新手完整学习

---

### 11. PROJECT_OVERVIEW.md
**语言**: 中文  
**内容**: 项目技术总览

**章节**:
- 项目架构
- 模块详解
- 数据流图
- 技术栈
- 参数配置指南
- 输出解读
- 常见场景
- 性能优化
- 改进方向

**适合**: 理解整体架构

---

### 12. FILES_INDEX.md
**语言**: 中文  
**内容**: 本文件，完整的文件清单

**用途**: 快速查找和理解每个文件

---

## 工作目录

### 13. checkpoints/
**用途**: 存储模型检查点

**内容**:
- `crossdocked_fullatom_cond.ckpt` - 预训练模型 (~500MB)
- 由setup_environment.sh自动下载

---

### 14. proteins/
**用途**: 存储蛋白PDB文件

**内容**:
- 用户提供的RE-CmeB蛋白PDB文件
- 例如: `RE-CmeB.pdb`

**要求**:
- 标准PDB格式
- 清除水分子
- 最好包含氢原子

---

### 15. results/
**用途**: 存储所有输出结果

**结构**:
```
results/RE-CmeB_iterative/
├── molecules/             # 生成的分子
├── models/                # 训练的模型
├── training_data/         # 训练数据
├── logs/                  # 日志和评分
├── uncertainty_analysis/  # 不确定性分析 ⭐ NEW
└── final_report.txt       # 最终报告
```

**子目录详情**:

#### molecules/
- `iteration_X_generated.sdf` - 每次迭代生成的所有分子
- `iteration_X_selected.sdf` - 每次迭代选中的1000个最优分子

#### models/
- `iteration_X_checkpoint.ckpt` - 每次迭代训练后的模型

#### training_data/
- `iteration_X_train.npz` - 每次迭代的训练数据

#### logs/
- `iteration_X_scores.csv` - 每个分子的详细评分（包含uncertainty和combined_score）
- `iteration_history.csv` - 迭代历史汇总
- 所有CSV文件可用Excel或Python打开

#### uncertainty_analysis/ ⭐ NEW
- `iteration_X_stats.json` - 每次迭代的选择统计
- `iteration_X_detailed_scores.csv` - 选中分子的详细评分
- `selection_history.csv` - 完整选择历史记录

---

## 快速查找

### 我想要...

**开始使用**:
→ 阅读 `快速开始指南.md`
→ 运行 `setup_environment.sh`

**理解原理**:
→ 阅读 `PROJECT_OVERVIEW.md`
→ 查看 `README.md`

**修改参数**:
→ 编辑 `run_example.sh`
→ 参考 `README.md` 参数说明部分

**调整评分**:
→ 编辑 `molecule_evaluator.py`
→ 修改 `self.weights` 字典

**检查环境**:
→ 运行 `python check_setup.py`

**查看结果**:
→ 查看 `results/*/final_report.txt`
→ 打开 `results/*/molecules/*.sdf`

**排查错误**:
→ 查看 `results/*/logs/`
→ 参考 `README.md` 故障排除部分

---

## 文件大小参考

| 文件 | 大小 | 说明 |
|------|------|------|
| iterative_generation.py | ~15KB | 主程序 |
| molecule_evaluator.py | ~8KB | 评估器 |
| prepare_training_data.py | ~8KB | 数据准备 |
| train_frozen.py | ~7KB | 冻结训练 |
| check_setup.py | ~6KB | 环境检查 |
| README.md | ~25KB | 详细文档 |
| crossdocked_fullatom_cond.ckpt | ~500MB | 预训练模型 |
| iteration_X_train.npz | ~10-50MB | 训练数据（取决于分子数） |
| iteration_X_checkpoint.ckpt | ~500MB | 训练后模型 |

**总磁盘需求**: 约20-50GB（30次迭代）

---

## 更新日志

**版本 1.2** (2025-10-25)
- ✓ 新增基于不确定性的智能选择策略
- ✓ 前期大胆探索，后期聚焦优化
- ✓ 避免过早陷入局部最优
- ✓ 预期多样性提升300%，最终得分提升8-15%
- ✓ 新增uncertainty_analysis输出目录
- ✓ 更新所有文档和脚本

**版本 1.1** (2025-10-24)
- ✓ 修复参数互斥性检查
- ✓ 修复3D构象生成
- ✓ 新增完整日志系统
- ✓ 新增训练恢复功能
- ✓ 内存优化和错误处理增强

**版本 1.0** (2025-10-23)
- ✓ 创建完整的迭代学习系统
- ✓ 实现多维度分子评估
- ✓ 实现冻结层训练策略
- ✓ 添加完整的文档和工具脚本
- ✓ 支持批量生成和评估

---

## 许可和引用

继承DiffSBDD的原始许可证。

引用：
```bibtex
@article{schneuing2024diffsbdd,
   title={Structure-based drug design with equivariant diffusion models},
   author={Schneuing, Arne and others},
   journal={Nature Computational Science},
   year={2024}
}
```

---

**最后更新**: 2025-10-24  
**维护者**: DiffSBDD Iterative Learning Team

