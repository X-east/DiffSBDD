"""
冻结层训练模块
固定EGNN底层，只训练上层参数
"""

import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import logging

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightning_modules import LigandPocketDDPM
from dataset import ProcessedLigandPocketDataset


def freeze_model_layers(model, freeze_bottom_layers=3, logger=None):
    """
    冻结模型的底层EGNN层
    
    Args:
        model: LightningModule模型
        freeze_bottom_layers: 要冻结的底层数量（默认3层）
        logger: 日志对象
    
    模型结构：
    - EGNN共有5层 (e_block_0 到 e_block_4)
    - 默认冻结前3层（保留通用知识），训练后2层（适应特定蛋白）
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*70)
    logger.info("冻结模型底层")
    logger.info("="*70)
    
    # 正确访问EGNN：model.ddpm.dynamics.egnn
    dynamics = model.ddpm.dynamics
    egnn = dynamics.egnn
    
    # 获取层数
    n_layers = egnn.n_layers
    logger.info(f"总EGNN层数: {n_layers}")
    logger.info(f"冻结策略: 冻结前 {freeze_bottom_layers} 层，训练后 {n_layers - freeze_bottom_layers} 层\n")
    
    frozen_params = 0
    trainable_params = 0
    
    # 冻结指定的EGNN底层
    for i in range(n_layers):
        block_name = f"e_block_{i}"
        if hasattr(egnn, '_modules') and block_name in egnn._modules:
            block = egnn._modules[block_name]
            
            if i < freeze_bottom_layers:
                # 冻结这一层
                for param in block.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                logger.info(f"  ✓ 层 {i} ({block_name}): 冻结")
            else:
                # 保持可训练
                for param in block.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
                logger.info(f"  ✓ 层 {i} ({block_name}): 可训练")
        else:
            logger.warning(f"  ✗ 警告: 未找到层 {block_name}")
    
    # 统计 encoder/decoder 参数（保持可训练，以适应新的蛋白特征）
    encoder_decoder_params = 0
    logger.info("\nEncoder/Decoder 模块:")
    for module_name in ['atom_encoder', 'atom_decoder', 
                        'residue_encoder', 'residue_decoder']:
        if hasattr(dynamics, module_name):
            module = getattr(dynamics, module_name)
            for param in module.parameters():
                param.requires_grad = True  # 保持可训练
                encoder_decoder_params += param.numel()
            logger.info(f"  ✓ {module_name}: 可训练")
    
    trainable_params += encoder_decoder_params
    
    # 统计 embedding 参数（保持可训练，以适应新的特征空间）
    embedding_params = 0
    logger.info("\nEmbedding 模块:")
    if hasattr(egnn, 'embedding'):
        for param in egnn.embedding.parameters():
            param.requires_grad = True
            embedding_params += param.numel()
        logger.info(f"  ✓ embedding: 可训练")
    
    if hasattr(egnn, 'embedding_out'):
        for param in egnn.embedding_out.parameters():
            param.requires_grad = True
            embedding_params += param.numel()
        logger.info(f"  ✓ embedding_out: 可训练")
    
    trainable_params += embedding_params
    
    # 输出统计信息
    logger.info(f"\n" + "-"*70)
    logger.info("参数统计:")
    logger.info("-"*70)
    logger.info(f"  冻结参数:     {frozen_params:>10,}")
    logger.info(f"  可训练参数:   {trainable_params:>10,}")
    logger.info(f"    - EGNN层:   {trainable_params - encoder_decoder_params - embedding_params:>10,}")
    logger.info(f"    - Encoder/Decoder: {encoder_decoder_params:>10,}")
    logger.info(f"    - Embedding: {embedding_params:>10,}")
    logger.info(f"  总参数:       {frozen_params + trainable_params:>10,}")
    logger.info(f"  可训练比例:   {100 * trainable_params / (frozen_params + trainable_params):>9.2f}%")
    logger.info("="*70 + "\n")
    
    return model


def train_with_frozen_layers(checkpoint, data_file, output_checkpoint,
                             n_epochs=50, freeze_bottom_layers=3,
                             batch_size=8, lr=1e-4, gpus=1, logger=None):
    """
    使用冻结层训练模型
    
    Args:
        checkpoint: 初始检查点路径
        data_file: 训练数据文件（npz格式）
        output_checkpoint: 输出检查点路径
        n_epochs: 训练轮数
        freeze_bottom_layers: 冻结的底层数量
        batch_size: 批次大小
        lr: 学习率
        gpus: GPU数量
        logger: 日志对象
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("开始冻结层训练")
    logger.info("="*70)
    logger.info(f"初始检查点: {checkpoint}")
    logger.info(f"训练数据: {data_file}")
    logger.info(f"输出检查点: {output_checkpoint}")
    logger.info(f"训练轮数: {n_epochs}")
    logger.info(f"冻结层数: {freeze_bottom_layers}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"学习率: {lr}")
    logger.info("="*70 + "\n")
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() and gpus > 0 else 'cpu'
    logger.info(f"使用设备: {device}")
    
    if device == 'cuda':
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        logger.info(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载模型
    logger.info("\n加载模型...")
    try:
        model = LigandPocketDDPM.load_from_checkpoint(checkpoint, map_location=device)
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise
    
    # 冻结底层
    model = freeze_model_layers(model, freeze_bottom_layers, logger)
    
    # 更新学习率
    model.lr = lr
    logger.info(f"学习率设置为: {lr}")
    
    # 加载数据
    logger.info(f"\n加载训练数据: {data_file}")
    
    # 检查数据文件是否存在
    if not Path(data_file).exists():
        raise FileNotFoundError(f"训练数据文件不存在: {data_file}")
    
    # 检查数据文件内容
    try:
        with np.load(data_file, allow_pickle=True) as data:
            logger.debug(f"数据文件包含的键: {list(data.keys())}")
            if 'lig_coords' in data:
                logger.info(f"配体坐标形状: {data['lig_coords'].shape}")
            if 'pocket_coords' in data:
                logger.info(f"口袋坐标形状: {data['pocket_coords'].shape}")
    except Exception as e:
        logger.error(f"读取数据文件失败: {e}")
        raise
    
    # 加载训练数据
    try:
        train_dataset = ProcessedLigandPocketDataset(
            data_file, 
            center=True,
            transform=model.data_transform if hasattr(model, 'data_transform') else None
        )
    except Exception as e:
        logger.error(f"创建数据集失败: {e}")
        raise
    
    logger.info(f"训练样本数: {len(train_dataset)}")
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
        pin_memory=(device == 'cuda')
    )
    
    logger.info(f"批次数: {len(train_loader)}")
    
    # 为了验证，我们使用训练集的一个小子集
    # 在实际应用中，你可能想要一个单独的验证集
    val_dataset = train_dataset
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=val_dataset.collate_fn,
        pin_memory=(device == 'cuda')
    )
    
    # 设置检查点回调
    output_path = Path(output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path.parent,
        filename=output_path.stem,
        monitor='loss/train',
        mode='min',
        save_last=False,
        save_top_k=1
    )
    
    # 早停回调
    early_stop_callback = EarlyStopping(
        monitor='loss/train',
        patience=10,
        mode='min',
        verbose=False
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=gpus if device == 'cuda' else 1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        check_val_every_n_epoch=5,
        logger=False  # 关闭pytorch lightning的默认日志
    )
    
    # 训练模型
    logger.info("\n开始训练...")
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        raise
    
    # 保存最终模型
    logger.info(f"\n保存最终模型到: {output_checkpoint}")
    trainer.save_checkpoint(output_checkpoint)
    
    logger.info("="*70)
    logger.info("训练完成！")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='冻结层训练')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='初始模型检查点')
    parser.add_argument('--data_file', type=str, required=True,
                        help='训练数据文件（npz格式）')
    parser.add_argument('--output_checkpoint', type=str, required=True,
                        help='输出检查点路径')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--freeze_layers', type=int, default=3,
                        help='冻结的底层数量（推荐3层，共5层EGNN）')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--gpus', type=int, default=1,
                        help='GPU数量')
    
    args = parser.parse_args()
    
    train_with_frozen_layers(
        checkpoint=args.checkpoint,
        data_file=args.data_file,
        output_checkpoint=args.output_checkpoint,
        n_epochs=args.n_epochs,
        freeze_bottom_layers=args.freeze_layers,
        batch_size=args.batch_size,
        lr=args.lr,
        gpus=args.gpus
    )

