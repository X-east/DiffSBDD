#!/bin/bash
# 环境设置脚本 - 准备运行迭代学习系统所需的环境

echo "=========================================="
echo "DiffSBDD 迭代学习系统 - 环境设置"
echo "=========================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null
then
    echo "错误: conda未安装"
    echo "请先安装Anaconda或Miniconda"
    exit 1
fi

# 创建conda环境
echo ""
echo "步骤 1/4: 创建conda环境..."
if conda env list | grep -q "^diffsbdd "; then
    echo "环境 'diffsbdd' 已存在"
    read -p "是否删除并重新创建？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n diffsbdd -y
        conda env create -f ../environment.yaml -n diffsbdd
    fi
else
    conda env create -f ../environment.yaml -n diffsbdd
fi

echo ""
echo "步骤 2/4: 创建必要的目录..."
mkdir -p checkpoints
mkdir -p proteins
mkdir -p results

# 下载预训练模型
echo ""
echo "步骤 3/4: 下载预训练模型..."
if [ ! -f "checkpoints/crossdocked_fullatom_cond.ckpt" ]; then
    echo "正在下载全原子条件模型 (~500MB)..."
    wget -P checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt
    echo "下载完成"
else
    echo "模型已存在，跳过下载"
fi

# 可选：安装smina（对接软件）
echo ""
echo "步骤 4/4: 安装对接软件 smina (可选)..."
read -p "是否安装 smina？这将用于对接打分评估 (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v smina.static &> /dev/null; then
        echo "smina 已安装"
    else
        echo "下载并安装 smina..."
        wget https://sourceforge.net/projects/smina/files/smina.static/download -O smina.static
        chmod +x smina.static
        
        # 尝试移动到系统路径
        if sudo mv smina.static /usr/local/bin/ 2>/dev/null; then
            echo "smina 已安装到 /usr/local/bin/"
        else
            echo "无法安装到系统路径，将保存在当前目录"
            echo "请手动添加到PATH或使用完整路径"
        fi
    fi
else
    echo "跳过 smina 安装（将无法使用对接打分功能）"
fi

echo ""
echo "=========================================="
echo "环境设置完成！"
echo "=========================================="
echo ""
echo "接下来的步骤："
echo "1. 激活环境: conda activate diffsbdd"
echo "2. 准备RE-CmeB蛋白PDB文件，放到 proteins/ 目录"
echo "3. 编辑 run_example.sh 配置参数"
echo "4. 运行: ./run_example.sh"
echo ""
echo "目录结构："
echo "  checkpoints/   - 预训练模型"
echo "  proteins/      - 蛋白PDB文件"
echo "  results/       - 输出结果"
echo ""
echo "=========================================="

