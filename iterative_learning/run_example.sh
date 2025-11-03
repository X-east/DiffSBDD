#!/bin/bash
# RE-CmeB蛋白迭代学习示例脚本

# =============================================================================
# 配置参数
# =============================================================================

# 模型和蛋白文件路径（请根据实际情况修改）
CHECKPOINT="./checkpoints/crossdocked_fullatom_cond.ckpt"  # 预训练模型路径
PDBFILE="./proteins/RE-CmeB.pdb"                           # RE-CmeB蛋白PDB文件
OUTPUT_DIR="./results/RE-CmeB_iterative"                   # 输出目录

# 口袋定义（二选一）
# 选项1：使用参考配体（如果PDB中包含配体）
REF_LIGAND="A:330"  # 格式：链:残基编号

# 选项2：使用残基列表定义口袋（如果没有参考配体）
# POCKET_IDS="A:1,A:2,A:3,A:4,A:5"

# 迭代参数
N_ITERATIONS=30      # 迭代次数
TRAIN_EPOCHS=50      # 每次迭代的训练轮数
FREEZE_LAYERS=3      # 冻结的EGNN底层数量（共5层，推荐冻结前3层）

# 训练参数
BATCH_SIZE=8         # 批次大小（根据GPU内存调整）
LR=0.0001           # 学习率

# 生成参数
TIMESTEPS=500        # 去噪步数（可选，默认使用训练值）

# 不确定性选择参数 ⭐ NEW
ALPHA_START=0.5      # 初始探索权重（前期大胆探索，推荐0.4-0.6）
ALPHA_END=0.03       # 最终探索权重（后期聚焦优化，推荐0.01-0.05）

# 评估参数
USE_DOCKING=""       # 是否使用对接打分（需要安装smina）
# USE_DOCKING="--use_docking"  # 取消注释以启用对接打分

# 恢复训练（如果需要从某次迭代继续）
RESUME_FROM=""       # 留空表示从头开始
# RESUME_FROM="--resume_from 5"  # 取消注释并设置迭代次数以从该迭代恢复

# =============================================================================
# 运行迭代学习
# =============================================================================

echo "=========================================="
echo "RE-CmeB蛋白迭代学习"
echo "=========================================="
echo "模型: $CHECKPOINT"
echo "蛋白: $PDBFILE"
echo "输出: $OUTPUT_DIR"
echo "迭代次数: $N_ITERATIONS"
if [ -n "$RESUME_FROM" ]; then
    echo "恢复训练: 是"
fi
echo "=========================================="
echo ""

# 检查文件是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: 模型检查点不存在: $CHECKPOINT"
    echo "请先下载预训练模型，例如："
    echo "wget -P checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt"
    exit 1
fi

if [ ! -f "$PDBFILE" ]; then
    echo "错误: 蛋白PDB文件不存在: $PDBFILE"
    echo "请提供RE-CmeB蛋白的PDB文件"
    exit 1
fi

# 运行迭代学习
python iterative_generation.py \
    --checkpoint "$CHECKPOINT" \
    --pdbfile "$PDBFILE" \
    --output_dir "$OUTPUT_DIR" \
    --ref_ligand "$REF_LIGAND" \
    --n_iterations $N_ITERATIONS \
    --train_epochs $TRAIN_EPOCHS \
    --freeze_layers $FREEZE_LAYERS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --timesteps $TIMESTEPS \
    --alpha_start $ALPHA_START \
    --alpha_end $ALPHA_END \
    $USE_DOCKING \
    $RESUME_FROM \
    --continue_on_error

# 如果使用残基列表定义口袋，请使用以下命令：
# python iterative_generation.py \
#     --checkpoint "$CHECKPOINT" \
#     --pdbfile "$PDBFILE" \
#     --output_dir "$OUTPUT_DIR" \
#     --pocket_ids "$POCKET_IDS" \
#     --n_iterations $N_ITERATIONS \
#     --train_epochs $TRAIN_EPOCHS \
#     --freeze_layers $FREEZE_LAYERS \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --alpha_start $ALPHA_START \
#     --alpha_end $ALPHA_END \
#     $USE_DOCKING \
#     --continue_on_error

echo ""
echo "=========================================="
echo "迭代学习完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="

