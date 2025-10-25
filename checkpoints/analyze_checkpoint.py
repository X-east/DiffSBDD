#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查点全面分析脚本
用于深入分析DiffSBDD模型检查点的结构、参数和配置
"""

import torch
import json
import numpy as np
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import sys
import os

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class CheckpointAnalyzer:
    """检查点分析器"""
    
    def __init__(self, checkpoint_path):
        """初始化分析器
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = None
        self.analysis_results = {}
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        print(f"[信息] 加载检查点: {self.checkpoint_path}")
        self.load_checkpoint()
        
    def load_checkpoint(self):
        """加载检查点文件"""
        try:
            # 在CPU上加载，避免GPU内存问题
            # PyTorch 2.6+ 需要 weights_only=False 来加载包含自定义类的检查点
            self.checkpoint = torch.load(
                self.checkpoint_path,
                map_location=torch.device('cpu'),
                weights_only=False
            )
            print(f"[成功] 检查点加载完成")
        except Exception as e:
            raise RuntimeError(f"加载检查点失败: {e}")
    
    def get_file_info(self):
        """获取文件基本信息"""
        file_size = self.checkpoint_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        info = {
            "文件名": self.checkpoint_path.name,
            "文件路径": str(self.checkpoint_path.absolute()),
            "文件大小": f"{file_size_mb:.2f} MB ({file_size:,} bytes)",
            "修改时间": datetime.fromtimestamp(
                self.checkpoint_path.stat().st_mtime
            ).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.analysis_results['file_info'] = info
        return info
    
    def analyze_checkpoint_structure(self):
        """分析检查点的整体结构"""
        structure = {
            "顶层键": list(self.checkpoint.keys()),
            "各键的类型": {k: type(v).__name__ for k, v in self.checkpoint.items()}
        }
        
        # 检查常见的键
        if 'epoch' in self.checkpoint:
            structure['训练轮数'] = self.checkpoint['epoch']
        if 'global_step' in self.checkpoint:
            structure['全局步数'] = self.checkpoint['global_step']
        
        self.analysis_results['structure'] = structure
        return structure
    
    def analyze_hyperparameters(self):
        """分析超参数"""
        if 'hyper_parameters' not in self.checkpoint:
            return {"错误": "未找到超参数"}
        
        hparams = self.checkpoint['hyper_parameters']
        
        # 提取关键超参数
        key_params = {}
        
        # 训练参数
        training_params = {
            'batch_size': hparams.get('batch_size'),
            'lr': hparams.get('lr'),
            'num_workers': hparams.get('num_workers'),
            'augment_noise': hparams.get('augment_noise'),
            'augment_rotation': hparams.get('augment_rotation'),
            'clip_grad': hparams.get('clip_grad'),
        }
        key_params['训练参数'] = training_params
        
        # EGNN参数
        if 'egnn_params' in hparams:
            egnn = hparams['egnn_params']
            egnn_params = {
                'n_layers': egnn.n_layers if hasattr(egnn, 'n_layers') else None,
                'hidden_nf': egnn.hidden_nf if hasattr(egnn, 'hidden_nf') else None,
                'attention': egnn.attention if hasattr(egnn, 'attention') else None,
                'normalization_factor': egnn.normalization_factor if hasattr(egnn, 'normalization_factor') else None,
                'aggregation_method': egnn.aggregation_method if hasattr(egnn, 'aggregation_method') else None,
            }
            key_params['EGNN参数'] = egnn_params
        
        # 扩散参数
        if 'diffusion_params' in hparams:
            diff = hparams['diffusion_params']
            diff_params = {
                'diffusion_steps': diff.diffusion_steps if hasattr(diff, 'diffusion_steps') else None,
                'diffusion_noise_schedule': diff.diffusion_noise_schedule if hasattr(diff, 'diffusion_noise_schedule') else None,
                'diffusion_loss_type': diff.diffusion_loss_type if hasattr(diff, 'diffusion_loss_type') else None,
            }
            key_params['扩散参数'] = diff_params
        
        # 模型模式
        key_params['模型模式'] = hparams.get('mode')
        key_params['口袋表示'] = hparams.get('pocket_representation')
        key_params['数据集'] = hparams.get('dataset')
        
        self.analysis_results['hyperparameters'] = key_params
        return key_params
    
    def count_parameters(self, state_dict):
        """统计参数数量"""
        total_params = 0
        trainable_params = 0
        
        param_details = {}
        
        for name, param in state_dict.items():
            num_params = param.numel()
            total_params += num_params
            trainable_params += num_params  # 从state_dict中的都是可训练的
            
            param_details[name] = {
                'shape': list(param.shape),
                'num_params': num_params,
                'dtype': str(param.dtype)
            }
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'details': param_details
        }
    
    def analyze_model_architecture(self):
        """分析模型架构"""
        if 'state_dict' not in self.checkpoint:
            return {"错误": "未找到state_dict"}
        
        state_dict = self.checkpoint['state_dict']
        
        # 统计参数
        param_stats = self.count_parameters(state_dict)
        
        # 按模块分组参数
        module_groups = self._group_parameters_by_module(state_dict)
        
        # 分析EGNN层
        egnn_layers = self._analyze_egnn_layers(state_dict)
        
        # 统计各模块参数数量
        module_param_counts = {}
        for module_name, params in module_groups.items():
            count = sum(p['num_params'] for p in params)
            module_param_counts[module_name] = count
        
        architecture = {
            '总参数数': f"{param_stats['total']:,}",
            '可训练参数数': f"{param_stats['trainable']:,}",
            '参数大小(MB)': f"{param_stats['total'] * 4 / (1024**2):.2f}",  # 假设float32
            '模块参数分布': module_param_counts,
            'EGNN层分析': egnn_layers,
            '参数详情': param_stats['details']
        }
        
        self.analysis_results['architecture'] = architecture
        return architecture
    
    def _group_parameters_by_module(self, state_dict):
        """按模块分组参数"""
        groups = {}
        
        for name, param in state_dict.items():
            # 移除前缀（如果有）
            clean_name = name.replace('model.', '').replace('_dynamics.', '')
            
            # 提取模块名（第一个点之前的部分）
            parts = clean_name.split('.')
            if len(parts) > 0:
                module_name = parts[0]
                if module_name not in groups:
                    groups[module_name] = []
                
                groups[module_name].append({
                    'name': clean_name,
                    'shape': list(param.shape),
                    'num_params': param.numel()
                })
        
        return groups
    
    def _analyze_egnn_layers(self, state_dict):
        """分析EGNN层结构"""
        egnn_info = {
            'num_layers': 0,
            'layers': []
        }
        
        # 查找EGNN层 (修复: 使用 e_block_ 而不是 e_blocks.)
        layer_nums = set()
        for name in state_dict.keys():
            if 'egnn' in name and '.e_block_' in name:
                # 提取层号
                try:
                    # 格式: ddpm.dynamics.egnn.e_block_0.gcl_0...
                    parts = name.split('.e_block_')
                    if len(parts) > 1:
                        layer_num = int(parts[1].split('.')[0])
                        layer_nums.add(layer_num)
                except:
                    continue
        
        egnn_info['num_layers'] = len(layer_nums)
        
        # 分析每一层
        for layer_num in sorted(layer_nums):
            layer_params = {}
            layer_param_count = 0
            
            for name, param in state_dict.items():
                # 修复: 使用 e_block_ 而不是 e_blocks.
                if f'.e_block_{layer_num}.' in name:
                    layer_params[name] = {
                        'shape': list(param.shape),
                        'num_params': param.numel()
                    }
                    layer_param_count += param.numel()
            
            egnn_info['layers'].append({
                'layer_num': layer_num,
                'num_params': layer_param_count,
                'num_submodules': len(layer_params)
            })
        
        return egnn_info
    
    def analyze_optimizer_state(self):
        """分析优化器状态"""
        if 'optimizer_states' not in self.checkpoint:
            return {"错误": "未找到优化器状态"}
        
        opt_states = self.checkpoint['optimizer_states']
        
        if len(opt_states) == 0:
            return {"错误": "优化器状态为空"}
        
        # 通常只有一个优化器
        opt_state = opt_states[0]
        
        info = {
            '状态键': list(opt_state.keys()),
            '参数组数量': len(opt_state.get('param_groups', [])),
        }
        
        # 获取学习率等信息
        if 'param_groups' in opt_state:
            param_groups = opt_state['param_groups']
            if len(param_groups) > 0:
                pg = param_groups[0]
                info['学习率'] = pg.get('lr')
                info['优化器类型'] = str(type(pg)).split('.')[-1].replace("'>", "")
                info['betas'] = pg.get('betas')
                info['eps'] = pg.get('eps')
                info['weight_decay'] = pg.get('weight_decay')
        
        # 统计状态信息
        if 'state' in opt_state:
            state = opt_state['state']
            info['状态参数数量'] = len(state)
            
            # 采样第一个参数的状态信息
            if len(state) > 0:
                first_key = list(state.keys())[0]
                first_state = state[first_key]
                info['状态包含的键'] = list(first_state.keys())
        
        self.analysis_results['optimizer'] = info
        return info
    
    def analyze_lr_scheduler(self):
        """分析学习率调度器"""
        if 'lr_schedulers' not in self.checkpoint:
            return {"信息": "未使用学习率调度器"}
        
        lr_schedulers = self.checkpoint['lr_schedulers']
        
        if len(lr_schedulers) == 0:
            return {"信息": "学习率调度器列表为空"}
        
        scheduler = lr_schedulers[0]
        
        info = {
            '调度器键': list(scheduler.keys()),
            '最后epoch': scheduler.get('last_epoch'),
            '_step_count': scheduler.get('_step_count'),
        }
        
        self.analysis_results['lr_scheduler'] = info
        return info
    
    def analyze_callbacks(self):
        """分析回调函数状态"""
        if 'callbacks' not in self.checkpoint:
            return {"信息": "未找到回调函数状态"}
        
        callbacks = self.checkpoint['callbacks']
        
        info = {
            '回调函数': list(callbacks.keys())
        }
        
        # 分析ModelCheckpoint回调
        for key in callbacks.keys():
            if 'ModelCheckpoint' in key:
                mc = callbacks[key]
                info['最佳模型得分'] = mc.get('best_model_score')
                info['最佳模型路径'] = mc.get('best_model_path')
        
        self.analysis_results['callbacks'] = info
        return info
    
    def generate_statistics(self):
        """生成统计摘要"""
        stats = {}
        
        # 从architecture中提取
        if 'architecture' in self.analysis_results:
            arch = self.analysis_results['architecture']
            stats['总参数数'] = arch.get('总参数数')
            stats['参数大小'] = arch.get('参数大小(MB)')
            
            # EGNN层数
            if 'EGNN层分析' in arch:
                egnn = arch['EGNN层分析']
                stats['EGNN层数'] = egnn.get('num_layers')
        
        # 从hyperparameters中提取
        if 'hyperparameters' in self.analysis_results:
            hp = self.analysis_results['hyperparameters']
            if 'EGNN参数' in hp:
                egnn_params = hp['EGNN参数']
                stats['隐藏层维度'] = egnn_params.get('hidden_nf')
                stats['注意力机制'] = egnn_params.get('attention')
            
            if '扩散参数' in hp:
                diff_params = hp['扩散参数']
                stats['扩散步数'] = diff_params.get('diffusion_steps')
        
        # 训练进度
        if 'structure' in self.analysis_results:
            struct = self.analysis_results['structure']
            stats['训练轮数'] = struct.get('训练轮数')
            stats['全局步数'] = struct.get('全局步数')
        
        self.analysis_results['statistics'] = stats
        return stats
    
    def analyze_all(self):
        """执行所有分析"""
        print("\n[步骤 1/8] 分析文件信息...")
        self.get_file_info()
        
        print("[步骤 2/8] 分析检查点结构...")
        self.analyze_checkpoint_structure()
        
        print("[步骤 3/8] 分析超参数...")
        self.analyze_hyperparameters()
        
        print("[步骤 4/8] 分析模型架构...")
        self.analyze_model_architecture()
        
        print("[步骤 5/8] 分析优化器状态...")
        self.analyze_optimizer_state()
        
        print("[步骤 6/8] 分析学习率调度器...")
        self.analyze_lr_scheduler()
        
        print("[步骤 7/8] 分析回调函数...")
        self.analyze_callbacks()
        
        print("[步骤 8/8] 生成统计摘要...")
        self.generate_statistics()
        
        print("\n[完成] 所有分析完成\n")
        
        return self.analysis_results
    
    def save_json_report(self, output_path):
        """保存JSON格式的分析报告"""
        # 转换不可序列化的对象
        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.dtype):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(self.analysis_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"[保存] JSON报告已保存到: {output_path}")
    
    def generate_markdown_report(self, output_path):
        """生成Markdown格式的详细分析报告"""
        md_lines = []
        
        # 标题
        md_lines.append("# DiffSBDD 模型检查点分析报告")
        md_lines.append("")
        md_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # 1. 执行摘要
        md_lines.append("## 📊 执行摘要")
        md_lines.append("")
        if 'statistics' in self.analysis_results:
            stats = self.analysis_results['statistics']
            md_lines.append("| 指标 | 值 |")
            md_lines.append("|------|------|")
            for key, value in stats.items():
                md_lines.append(f"| {key} | {value} |")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # 2. 文件信息
        md_lines.append("## 📁 文件信息")
        md_lines.append("")
        if 'file_info' in self.analysis_results:
            info = self.analysis_results['file_info']
            for key, value in info.items():
                md_lines.append(f"- **{key}**: `{value}`")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # 3. 检查点结构
        md_lines.append("## 🏗️ 检查点结构")
        md_lines.append("")
        if 'structure' in self.analysis_results:
            struct = self.analysis_results['structure']
            
            md_lines.append("### 顶层键")
            md_lines.append("```")
            for key in struct.get('顶层键', []):
                md_lines.append(f"  - {key}")
            md_lines.append("```")
            md_lines.append("")
            
            if '训练轮数' in struct:
                md_lines.append(f"- **训练轮数**: {struct['训练轮数']}")
            if '全局步数' in struct:
                md_lines.append(f"- **全局步数**: {struct['全局步数']}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # 4. 超参数配置
        md_lines.append("## ⚙️ 超参数配置")
        md_lines.append("")
        if 'hyperparameters' in self.analysis_results:
            hp = self.analysis_results['hyperparameters']
            
            for category, params in hp.items():
                if isinstance(params, dict):
                    md_lines.append(f"### {category}")
                    md_lines.append("")
                    md_lines.append("| 参数 | 值 |")
                    md_lines.append("|------|------|")
                    for key, value in params.items():
                        md_lines.append(f"| {key} | {value} |")
                    md_lines.append("")
                else:
                    md_lines.append(f"- **{category}**: {params}")
                    md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # 5. 模型架构
        md_lines.append("## 🧠 模型架构分析")
        md_lines.append("")
        if 'architecture' in self.analysis_results:
            arch = self.analysis_results['architecture']
            
            # 总览
            md_lines.append("### 架构总览")
            md_lines.append("")
            md_lines.append("| 指标 | 值 |")
            md_lines.append("|------|------|")
            md_lines.append(f"| 总参数数 | {arch.get('总参数数', 'N/A')} |")
            md_lines.append(f"| 可训练参数数 | {arch.get('可训练参数数', 'N/A')} |")
            md_lines.append(f"| 参数大小 | {arch.get('参数大小(MB)', 'N/A')} MB |")
            md_lines.append("")
            
            # 模块参数分布
            if '模块参数分布' in arch:
                md_lines.append("### 模块参数分布")
                md_lines.append("")
                md_lines.append("| 模块 | 参数数量 | 占比 |")
                md_lines.append("|------|----------|------|")
                
                total = sum(arch['模块参数分布'].values())
                for module, count in sorted(
                    arch['模块参数分布'].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    percentage = (count / total * 100) if total > 0 else 0
                    md_lines.append(f"| {module} | {count:,} | {percentage:.2f}% |")
                md_lines.append("")
            
            # EGNN层分析
            if 'EGNN层分析' in arch:
                egnn = arch['EGNN层分析']
                md_lines.append("### EGNN 层结构")
                md_lines.append("")
                md_lines.append(f"**总层数**: {egnn.get('num_layers', 0)}")
                md_lines.append("")
                
                if 'layers' in egnn and len(egnn['layers']) > 0:
                    md_lines.append("| 层编号 | 参数数量 | 子模块数 |")
                    md_lines.append("|--------|----------|----------|")
                    for layer in egnn['layers']:
                        md_lines.append(
                            f"| Layer {layer['layer_num']} | "
                            f"{layer['num_params']:,} | "
                            f"{layer['num_submodules']} |"
                        )
                    md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # 6. 优化器状态
        md_lines.append("## 🎯 优化器状态")
        md_lines.append("")
        if 'optimizer' in self.analysis_results:
            opt = self.analysis_results['optimizer']
            md_lines.append("| 参数 | 值 |")
            md_lines.append("|------|------|")
            for key, value in opt.items():
                if key not in ['状态键', '状态包含的键']:
                    md_lines.append(f"| {key} | {value} |")
            md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # 7. 学习率调度器
        md_lines.append("## 📈 学习率调度器")
        md_lines.append("")
        if 'lr_scheduler' in self.analysis_results:
            lr_sched = self.analysis_results['lr_scheduler']
            for key, value in lr_sched.items():
                md_lines.append(f"- **{key}**: {value}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # 8. 回调函数
        md_lines.append("## 🔔 回调函数")
        md_lines.append("")
        if 'callbacks' in self.analysis_results:
            cb = self.analysis_results['callbacks']
            for key, value in cb.items():
                md_lines.append(f"- **{key}**: {value}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # 9. 详细参数列表（可选，太长的话可以注释掉）
        md_lines.append("## 📋 详细参数列表")
        md_lines.append("")
        md_lines.append("<details>")
        md_lines.append("<summary>点击展开完整参数列表（可能很长）</summary>")
        md_lines.append("")
        
        if 'architecture' in self.analysis_results:
            arch = self.analysis_results['architecture']
            if '参数详情' in arch:
                md_lines.append("| 参数名 | 形状 | 参数数 | 数据类型 |")
                md_lines.append("|--------|------|--------|----------|")
                
                # 只显示前100个参数，避免太长
                param_details = arch['参数详情']
                for i, (name, details) in enumerate(param_details.items()):
                    if i >= 100:
                        md_lines.append(f"| ... | ... | ... | ... |")
                        md_lines.append(f"| *省略剩余 {len(param_details) - 100} 个参数* | | | |")
                        break
                    md_lines.append(
                        f"| {name} | "
                        f"{details['shape']} | "
                        f"{details['num_params']:,} | "
                        f"{details['dtype']} |"
                    )
        
        md_lines.append("")
        md_lines.append("</details>")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # 10. 结论与建议
        md_lines.append("## 💡 结论与建议")
        md_lines.append("")
        
        # 基于分析结果给出建议
        if 'statistics' in self.analysis_results:
            stats = self.analysis_results['statistics']
            
            md_lines.append("### 模型特征")
            md_lines.append("")
            
            # 参数量评估
            total_params_str = stats.get('总参数数', '0')
            total_params = int(total_params_str.replace(',', '')) if total_params_str != 'N/A' else 0
            
            if total_params < 1_000_000:
                md_lines.append(f"- ✅ **轻量级模型**: 约 {total_params/1000:.1f}K 参数，适合快速训练和推理")
            elif total_params < 10_000_000:
                md_lines.append(f"- ✅ **中等规模模型**: 约 {total_params/1_000_000:.2f}M 参数，平衡性能与效率")
            else:
                md_lines.append(f"- ⚠️ **大型模型**: 约 {total_params/1_000_000:.2f}M 参数，需要较多计算资源")
            
            md_lines.append("")
            
            # EGNN层数评估
            if 'EGNN层数' in stats:
                num_layers = stats['EGNN层数']
                md_lines.append(f"- **EGNN深度**: {num_layers} 层")
                if num_layers <= 4:
                    md_lines.append("  - 较浅的网络，训练快速，适合迭代学习")
                elif num_layers <= 8:
                    md_lines.append("  - 中等深度，良好的表达能力")
                else:
                    md_lines.append("  - 深层网络，强大的表达能力但训练较慢")
            
            md_lines.append("")
        
        md_lines.append("### 迭代学习建议")
        md_lines.append("")
        md_lines.append("基于此检查点进行迭代学习时的建议：")
        md_lines.append("")
        md_lines.append("1. **冻结策略**: 建议冻结前 2-4 层，训练后 2 层")
        md_lines.append("   - 保留底层通用化学知识")
        md_lines.append("   - 适应特定蛋白的结合模式")
        md_lines.append("")
        md_lines.append("2. **学习率设置**: 建议使用较小的学习率 (1e-4 到 1e-5)")
        md_lines.append("   - 避免破坏预训练权重")
        md_lines.append("   - 实现稳定的微调")
        md_lines.append("")
        md_lines.append("3. **批次大小**: 根据GPU显存调整")
        md_lines.append("   - 8GB GPU: batch_size = 4-8")
        md_lines.append("   - 12GB GPU: batch_size = 8-16")
        md_lines.append("   - 24GB GPU: batch_size = 16-32")
        md_lines.append("")
        
        # 结尾
        md_lines.append("---")
        md_lines.append("")
        md_lines.append("**报告生成工具**: `analyze_checkpoint.py`")
        md_lines.append("")
        md_lines.append(f"**检查点**: `{self.checkpoint_path.name}`")
        md_lines.append("")
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        
        print(f"[保存] Markdown报告已保存到: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DiffSBDD检查点全面分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python analyze_checkpoint.py crossdocked_fullatom_cond.ckpt
  python analyze_checkpoint.py crossdocked_fullatom_cond.ckpt -o my_analysis
        """
    )
    
    parser.add_argument(
        'checkpoint',
        type=str,
        help='检查点文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出文件名前缀（默认使用检查点文件名）'
    )
    
    args = parser.parse_args()
    
    # 确定输出文件名
    if args.output is None:
        checkpoint_name = Path(args.checkpoint).stem
        output_prefix = f"{checkpoint_name}_analysis"
    else:
        output_prefix = args.output
    
    # 创建分析器
    try:
        analyzer = CheckpointAnalyzer(args.checkpoint)
        
        # 执行所有分析
        results = analyzer.analyze_all()
        
        # 保存报告
        json_path = f"{output_prefix}.json"
        md_path = f"{output_prefix}.md"
        
        analyzer.save_json_report(json_path)
        analyzer.generate_markdown_report(md_path)
        
        print("\n" + "="*60)
        print("[OK] 分析完成！")
        print("="*60)
        print(f"\n生成的文件:")
        print(f"  [JSON] JSON报告: {json_path}")
        print(f"  [MD] Markdown报告: {md_path}")
        print(f"\n请查看Markdown报告以获取详细分析结果。\n")
        
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

