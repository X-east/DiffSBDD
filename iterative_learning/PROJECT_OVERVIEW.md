# RE-CmeB蛋白迭代学习项目总览

## 项目架构

```
DiffSBDD-main/
└── iterative_learning/              # 新建的迭代学习系统
    ├── iterative_generation.py      # [核心] 主程序
    ├── molecule_evaluator.py        # [核心] 分子评估器
    ├── prepare_training_data.py     # [核心] 数据准备器
    ├── train_frozen.py              # [核心] 冻结训练模块
    ├── run_example.sh               # 运行脚本
    ├── setup_environment.sh         # 环境设置脚本
    ├── README.md                    # 英文详细文档
    ├── 使用说明.txt                 # 中文快速指南
    └── PROJECT_OVERVIEW.md          # 本文件
```

## 模块说明

### 1. iterative_generation.py (主程序)

**功能**: 协调整个迭代学习流程

**核心类**: `IterativeLearning`

**主要方法**:
- `generate_molecules()`: 使用模型生成分子
- `select_best_molecules()`: 根据评分选择最优分子
- `train_on_selected_molecules()`: 用选中分子训练模型
- `run_iteration()`: 运行单次迭代
- `run()`: 运行完整的30次迭代

**工作流程**:
```
初始化 → 加载模型 → 迭代循环 → 生成报告
         ↓
         每次迭代:
         1. 生成分子
         2. 评估分子
         3. 选择优秀分子
         4. 训练模型
```

### 2. molecule_evaluator.py (评估器)

**功能**: 评估分子的多维度性质

**核心类**: `MoleculeEvaluator`

**评估指标**:
- QED (类药性): [0, 1]，越大越好
- SA (合成可及性): [0, 1]，越大越容易合成
- LogP (亲脂性): 理想范围[-2, 5]
- Lipinski (类药五规则): [0, 5]，满足的规则数
- Docking Score (对接打分): 负值，越负越好

**评分公式**:
```
综合得分 = 0.25×QED_norm + 0.25×SA_norm + 0.15×LogP_norm 
         + 0.15×Lipinski_norm + 0.20×Docking_norm
```

### 3. prepare_training_data.py (数据准备)

**功能**: 将RDKit分子转换为DiffSBDD训练格式

**核心类**: `DataPreparer`

**转换流程**:
```
RDKit分子 → 提取坐标和原子类型 → 编码 → 与口袋组合 → NPZ格式
```

**输出格式** (NPZ文件):
- `lig_coords`: 配体坐标
- `lig_one_hot`: 配体原子类型（one-hot编码）
- `lig_mask`: 配体批次掩码
- `pocket_coords`: 口袋坐标
- `pocket_one_hot`: 口袋类型（one-hot编码）
- `pocket_mask`: 口袋批次掩码

### 4. train_frozen.py (冻结训练)

**功能**: 固定EGNN底层，只训练上层

**核心函数**: 
- `freeze_model_layers()`: 冻结指定层
- `train_with_frozen_layers()`: 执行训练

**冻结策略**:
```
EGNN结构（共5层）:
├── Layer 0  [冻结] ─┐
├── Layer 1  [冻结]  ├─ 保持预训练通用知识
├── Layer 2  [冻结] ─┘
├── Layer 3  [训练] ─┐
└── Layer 4  [训练] ─┘ 适应特定蛋白
```

**优势**:
- 训练速度快（参数少）
- 避免过拟合
- 保留基础能力

## 数据流图

```
┌─────────────┐
│ 预训练模型  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  迭代 1                              │
│  1. 生成2000个分子                   │
│  2. 评估（QED/SA/LogP/Lipinski）    │
│  3. 选择1000个最优                   │
│  4. 准备训练数据（NPZ）              │
│  5. 冻结训练（固定底3层）            │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  迭代 2-30                           │
│  1. 生成1000个新分子                 │
│  2. 与上次1000个合并                 │
│  3. 评估2000个分子                   │
│  4. 选择1000个最优                   │
│  5. 准备训练数据                     │
│  6. 冻结训练                         │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│ 最终报告    │
│ 最优分子    │
└─────────────┘
```

## 技术栈

### 核心依赖
- **PyTorch**: 深度学习框架
- **PyTorch Lightning**: 训练框架
- **RDKit**: 化学信息学库
- **BioPython**: 蛋白结构处理
- **NumPy/Pandas**: 数据处理

### 可选依赖
- **Smina**: 分子对接（对接打分）
- **Wandb**: 实验跟踪（如需）

## 参数配置指南

### 计算资源配置

| GPU显存 | batch_size | train_epochs | 预计单次迭代时间 |
|---------|-----------|--------------|-----------------|
| 8GB     | 4         | 30           | ~2小时          |
| 12GB    | 8         | 50           | ~3小时          |
| 24GB    | 16        | 50           | ~2小时          |

### 质量优先配置
```bash
--train_epochs 100      # 更多训练
--freeze_layers 3       # 训练更多层
--lr 5e-5              # 更小学习率
--use_docking          # 使用对接打分
```

### 速度优先配置
```bash
--train_epochs 30       # 较少训练
--freeze_layers 5       # 冻结更多层
--batch_size 16        # 更大批次（需要GPU）
```

### 平衡配置（推荐）
```bash
--train_epochs 50
--freeze_layers 3
--batch_size 8
--lr 1e-4
```

## 输出解读

### iteration_history.csv
记录每次迭代的统计信息，关键列：
- `iteration`: 迭代次数
- `avg_qed`: 平均QED（关注趋势）
- `avg_sa`: 平均合成可及性（越高越好）
- `avg_docking_score`: 平均对接分数（越负越好）
- `avg_综合得分`: 平均综合得分（主要指标）

### iteration_X_scores.csv
每个分子的详细评分，用于：
- 筛选特定性质的分子
- 分析分子分布
- 识别优秀样本

### final_report.txt
汇总报告，包含：
- 整体统计
- 性能趋势
- 最优分子位置

## 常见使用场景

### 场景1: 首次运行
```bash
# 1. 设置环境
./setup_environment.sh

# 2. 准备PDB文件
cp /path/to/RE-CmeB.pdb proteins/

# 3. 编辑配置
nano run_example.sh

# 4. 运行
./run_example.sh
```

### 场景2: 继续未完成的迭代
如果中断，可以修改代码从特定迭代继续：

```python
# 在iterative_generation.py的run()方法中
for iteration in range(START_ITERATION, self.args.n_iterations + 1):
    # 设置START_ITERATION为你想继续的迭代
```

### 场景3: 只评估不训练
```python
# 修改run_iteration()跳过训练部分
if iteration < self.args.n_iterations:
    # checkpoint_path = self.train_on_selected_molecules(...)
    pass  # 跳过训练
```

### 场景4: 使用自定义评分
编辑 `molecule_evaluator.py`:
```python
self.weights = {
    'QED': 0.3,      # 增加类药性权重
    'SA': 0.2,       # 降低合成可及性权重
    'LogP': 0.1,
    'Lipinski': 0.1,
    'Docking': 0.3   # 增加对接权重
}
```

## 性能优化建议

### 内存优化
1. 减小批次大小
2. 使用梯度累积
3. 清理中间变量

### 速度优化
1. 使用多GPU（修改gpus参数）
2. 增加num_workers
3. 使用混合精度训练（需修改代码）

### 质量优化
1. 增加训练轮数
2. 使用对接打分
3. 调整评分权重
4. 使用更大的模型

## 故障排查清单

- [ ] 检查conda环境是否激活
- [ ] 检查PDB文件格式是否正确
- [ ] 检查口袋定义是否合理
- [ ] 检查GPU是否可用（nvidia-smi）
- [ ] 检查磁盘空间是否充足
- [ ] 检查依赖包是否完整安装
- [ ] 查看日志文件了解详细错误

## 下一步改进方向

### 功能扩展
1. 支持多蛋白并行优化
2. 添加更多评估指标（如PAINS过滤）
3. 集成可视化界面
4. 支持约束生成（如保留特定基团）

### 性能改进
1. 实现分布式训练
2. 优化数据加载
3. 缓存中间结果
4. 使用更高效的对接软件

### 算法优化
1. 动态调整冻结策略
2. 实现主动学习
3. 多目标优化
4. 集成强化学习

## 参考资料

- [DiffSBDD原始论文](https://doi.org/10.1038/s43588-024-00737-x)
- [DiffSBDD GitHub](https://github.com/arneschneuing/DiffSBDD)
- [RDKit文档](https://www.rdkit.org/docs/)
- [PyTorch Lightning文档](https://pytorch-lightning.readthedocs.io/)

---

**版本**: 1.0
**更新日期**: 2025-10-24

