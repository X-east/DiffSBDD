# DiffSBDD 模型检查点分析报告

**生成时间**: 2025-10-25 11:02:01

---

## 📊 执行摘要

| 指标 | 值 |
|------|------|
| 总参数数 | 1,006,560 |
| 参数大小 | 3.84 |
| EGNN层数 | 5 |
| 隐藏层维度 | 128 |
| 注意力机制 | True |
| 扩散步数 | 500 |
| 训练轮数 | 999 |
| 全局步数 | 1562000 |

---

## 📁 文件信息

- **文件名**: `crossdocked_fullatom_cond.ckpt`
- **文件路径**: `D:\Desktop\DiffSBDD-main\checkpoints\crossdocked_fullatom_cond.ckpt`
- **文件大小**: `17.03 MB (17,861,341 bytes)`
- **修改时间**: `2025-10-25 09:14:24`

---

## 🏗️ 检查点结构

### 顶层键
```
  - epoch
  - global_step
  - pytorch-lightning_version
  - state_dict
  - loops
  - callbacks
  - optimizer_states
  - lr_schedulers
  - hparams_name
  - hyper_parameters
```

- **训练轮数**: 999
- **全局步数**: 1562000

---

## ⚙️ 超参数配置

### 训练参数

| 参数 | 值 |
|------|------|
| batch_size | 16 |
| lr | 0.001 |
| num_workers | 0 |
| augment_noise | 0 |
| augment_rotation | False |
| clip_grad | True |

### EGNN参数

| 参数 | 值 |
|------|------|
| n_layers | 5 |
| hidden_nf | 128 |
| attention | True |
| normalization_factor | 100 |
| aggregation_method | sum |

### 扩散参数

| 参数 | 值 |
|------|------|
| diffusion_steps | 500 |
| diffusion_noise_schedule | polynomial_2 |
| diffusion_loss_type | l2 |

- **模型模式**: pocket_conditioning

- **口袋表示**: full-atom

- **数据集**: crossdock

---

## 🧠 模型架构分析

### 架构总览

| 指标 | 值 |
|------|------|
| 总参数数 | 1,006,560 |
| 可训练参数数 | 1,006,560 |
| 参数大小 | 3.84 MB |

### 模块参数分布

| 模块 | 参数数量 | 占比 |
|------|----------|------|
| ddpm | 1,006,560 | 100.00% |

### EGNN 层结构

**总层数**: 5

| 层编号 | 参数数量 | 子模块数 |
|--------|----------|----------|
| Layer 0 | 198,785 | 20 |
| Layer 1 | 198,785 | 20 |
| Layer 2 | 198,785 | 20 |
| Layer 3 | 198,785 | 20 |
| Layer 4 | 198,785 | 20 |

---

## 🎯 优化器状态

| 参数 | 值 |
|------|------|
| 参数组数量 | 1 |
| 学习率 | 0.001 |
| 优化器类型 | <class 'dict |
| betas | (0.9, 0.999) |
| eps | 1e-08 |
| weight_decay | 1e-12 |
| 状态参数数量 | 111 |

---

## 📈 学习率调度器


---

## 🔔 回调函数

- **回调函数**: ["ModelCheckpoint{'monitor': 'loss/val', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None, 'save_on_train_epoch_end': True}"]
- **最佳模型得分**: -20.816247940063477
- **最佳模型路径**: /mnt/beegfs/bulk/mirror/yuanqi/DiffSBDD_dev/ligand-pocket-ddpm/training_logs/conditional-full-crossdock-egnn-nf128-jointnf32-n_layers5-lr1e-3-steps500/checkpoints/best-model-epoch=epoch=987.ckpt

---

## 📋 详细参数列表

<details>
<summary>点击展开完整参数列表（可能很长）</summary>

| 参数名 | 形状 | 参数数 | 数据类型 |
|--------|------|--------|----------|
| ddpm.buffer | [1] | 1 | torch.float32 |
| ddpm.gamma.gamma | [501] | 501 | torch.float32 |
| ddpm.dynamics.atom_encoder.0.weight | [20, 10] | 200 | torch.float32 |
| ddpm.dynamics.atom_encoder.0.bias | [20] | 20 | torch.float32 |
| ddpm.dynamics.atom_encoder.2.weight | [32, 20] | 640 | torch.float32 |
| ddpm.dynamics.atom_encoder.2.bias | [32] | 32 | torch.float32 |
| ddpm.dynamics.atom_decoder.0.weight | [20, 32] | 640 | torch.float32 |
| ddpm.dynamics.atom_decoder.0.bias | [20] | 20 | torch.float32 |
| ddpm.dynamics.atom_decoder.2.weight | [10, 20] | 200 | torch.float32 |
| ddpm.dynamics.atom_decoder.2.bias | [10] | 10 | torch.float32 |
| ddpm.dynamics.residue_encoder.0.weight | [20, 10] | 200 | torch.float32 |
| ddpm.dynamics.residue_encoder.0.bias | [20] | 20 | torch.float32 |
| ddpm.dynamics.residue_encoder.2.weight | [32, 20] | 640 | torch.float32 |
| ddpm.dynamics.residue_encoder.2.bias | [32] | 32 | torch.float32 |
| ddpm.dynamics.residue_decoder.0.weight | [20, 32] | 640 | torch.float32 |
| ddpm.dynamics.residue_decoder.0.bias | [20] | 20 | torch.float32 |
| ddpm.dynamics.residue_decoder.2.weight | [10, 20] | 200 | torch.float32 |
| ddpm.dynamics.residue_decoder.2.bias | [10] | 10 | torch.float32 |
| ddpm.dynamics.egnn.embedding.weight | [128, 33] | 4,224 | torch.float32 |
| ddpm.dynamics.egnn.embedding.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.embedding_out.weight | [33, 128] | 4,224 | torch.float32 |
| ddpm.dynamics.egnn.embedding_out.bias | [33] | 33 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_0.edge_mlp.0.weight | [128, 258] | 33,024 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_0.edge_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_0.edge_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_0.edge_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_0.node_mlp.0.weight | [128, 256] | 32,768 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_0.node_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_0.node_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_0.node_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_0.att_mlp.0.weight | [1, 128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_0.att_mlp.0.bias | [1] | 1 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_equiv.coord_mlp.0.weight | [128, 258] | 33,024 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_equiv.coord_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_equiv.coord_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_equiv.coord_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_equiv.coord_mlp.4.weight | [1, 128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_equiv.cross_product_mlp.0.weight | [128, 258] | 33,024 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_equiv.cross_product_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_equiv.cross_product_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_equiv.cross_product_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_0.gcl_equiv.cross_product_mlp.4.weight | [1, 128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_0.edge_mlp.0.weight | [128, 258] | 33,024 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_0.edge_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_0.edge_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_0.edge_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_0.node_mlp.0.weight | [128, 256] | 32,768 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_0.node_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_0.node_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_0.node_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_0.att_mlp.0.weight | [1, 128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_0.att_mlp.0.bias | [1] | 1 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_equiv.coord_mlp.0.weight | [128, 258] | 33,024 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_equiv.coord_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_equiv.coord_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_equiv.coord_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_equiv.coord_mlp.4.weight | [1, 128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_equiv.cross_product_mlp.0.weight | [128, 258] | 33,024 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_equiv.cross_product_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_equiv.cross_product_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_equiv.cross_product_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_1.gcl_equiv.cross_product_mlp.4.weight | [1, 128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_0.edge_mlp.0.weight | [128, 258] | 33,024 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_0.edge_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_0.edge_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_0.edge_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_0.node_mlp.0.weight | [128, 256] | 32,768 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_0.node_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_0.node_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_0.node_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_0.att_mlp.0.weight | [1, 128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_0.att_mlp.0.bias | [1] | 1 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_equiv.coord_mlp.0.weight | [128, 258] | 33,024 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_equiv.coord_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_equiv.coord_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_equiv.coord_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_equiv.coord_mlp.4.weight | [1, 128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_equiv.cross_product_mlp.0.weight | [128, 258] | 33,024 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_equiv.cross_product_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_equiv.cross_product_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_equiv.cross_product_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_2.gcl_equiv.cross_product_mlp.4.weight | [1, 128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_0.edge_mlp.0.weight | [128, 258] | 33,024 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_0.edge_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_0.edge_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_0.edge_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_0.node_mlp.0.weight | [128, 256] | 32,768 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_0.node_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_0.node_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_0.node_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_0.att_mlp.0.weight | [1, 128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_0.att_mlp.0.bias | [1] | 1 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_equiv.coord_mlp.0.weight | [128, 258] | 33,024 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_equiv.coord_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_equiv.coord_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_equiv.coord_mlp.2.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_equiv.coord_mlp.4.weight | [1, 128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_equiv.cross_product_mlp.0.weight | [128, 258] | 33,024 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_equiv.cross_product_mlp.0.bias | [128] | 128 | torch.float32 |
| ddpm.dynamics.egnn.e_block_3.gcl_equiv.cross_product_mlp.2.weight | [128, 128] | 16,384 | torch.float32 |
| ... | ... | ... | ... |
| *省略剩余 22 个参数* | | | |

</details>

---

## 💡 结论与建议

### 模型特征

- ✅ **中等规模模型**: 约 1.01M 参数，平衡性能与效率

- **EGNN深度**: 5 层
  - 中等深度，良好的表达能力

### 迭代学习建议

基于此检查点进行迭代学习时的建议：

1. **冻结策略**: ⭐ 推荐冻结前 3 层，训练后 2 层
   - 冻结 Layer 0-2：保留底层通用化学知识
   - 训练 Layer 3-4：适应特定蛋白的结合模式
   - 可训练参数约 40%，平衡性能与速度

2. **学习率设置**: 建议使用较小的学习率 (1e-4 到 1e-5)
   - 避免破坏预训练权重
   - 实现稳定的微调
   - 推荐起始值: 1e-4

3. **批次大小**: 根据GPU显存调整（模型轻量，可用较大batch）
   - 8GB GPU: batch_size = 8-12
   - 12GB GPU: batch_size = 16-24
   - 24GB GPU: batch_size = 32+

---

**报告生成工具**: `analyze_checkpoint.py`

**检查点**: `crossdocked_fullatom_cond.ckpt`
