"""
迭代学习主程序 - 针对RE-CmeB蛋白的优化分子生成
用途：通过迭代学习生成对特定蛋白效果更好的分子

流程：
1. 初始生成2000个分子
2. 评估并选出1000个优秀分子
3. 用这1000个分子重新训练模型（固定底层，只训练上层）
4. 生成1000个新分子，与之前的1000个合并成2000个
5. 从2000个中选出1000个优秀分子
6. 重复30次迭代
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import json
import gc

# 添加父目录到路径，以便导入DiffSBDD模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightning_modules import LigandPocketDDPM
from molecule_evaluator import MoleculeEvaluator
from prepare_training_data import prepare_iterative_training_data
from train_frozen import train_with_frozen_layers
from uncertainty_selector import UncertaintyBasedSelector
import utils


def setup_logger(output_dir, resume_from=None):
    """
    设置日志系统
    
    Args:
        output_dir: 输出目录
        resume_from: 如果恢复训练，指定从哪次迭代开始
    
    Returns:
        logger对象
    """
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('IterativeLearning')
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件handler
    if resume_from:
        log_file = log_dir / f"training_resume_from_{resume_from}.log"
    else:
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    logger.info(f"日志文件: {log_file}")
    
    return logger


class IterativeLearning:
    """迭代学习主类"""
    
    def __init__(self, args, logger=None):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = logger or logging.getLogger('IterativeLearning')
        
        # 验证参数互斥性
        if not ((args.pocket_ids is None) ^ (args.ref_ligand is None)):
            raise ValueError("必须指定且仅指定 --pocket_ids 或 --ref_ligand 之一")
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建子目录
        self.molecules_dir = self.output_dir / "molecules"
        self.models_dir = self.output_dir / "models"
        self.data_dir = self.output_dir / "training_data"
        self.logs_dir = self.output_dir / "logs"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        
        for d in [self.molecules_dir, self.models_dir, self.data_dir, 
                  self.logs_dir, self.checkpoints_dir]:
            d.mkdir(exist_ok=True)
        
        # 加载或恢复状态
        self.state_file = self.checkpoints_dir / "training_state.json"
        self.current_iteration = 1
        self.iteration_history = []
        
        if args.resume_from is not None:
            self._load_state(args.resume_from)
        
        # 加载初始模型
        self.logger.info(f"从 {args.checkpoint} 加载初始模型...")
        self.initial_model = LigandPocketDDPM.load_from_checkpoint(
            args.checkpoint, map_location=self.device
        )
        self.initial_model = self.initial_model.to(self.device)
        self.logger.info(f"模型已加载到设备: {self.device}")
        
        # 初始化评估器
        self.evaluator = MoleculeEvaluator(
            pdb_file=args.pdbfile,
            use_docking=args.use_docking,
            logger=self.logger
        )
        
        # 初始化不确定性选择器
        self.uncertainty_selector = UncertaintyBasedSelector(
            output_dir=self.output_dir,
            alpha_start=args.alpha_start if hasattr(args, 'alpha_start') else 0.5,
            alpha_end=args.alpha_end if hasattr(args, 'alpha_end') else 0.03,
            n_iterations=args.n_iterations,
            logger=self.logger
        )
        
        # 当前分子池（用于内存管理）
        self.current_molecules_pool = []
    
    def _save_state(self, iteration):
        """保存当前训练状态"""
        state = {
            'iteration': iteration,
            'iteration_history': self.iteration_history,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        self.logger.info(f"训练状态已保存: 迭代 {iteration}")
    
    def _load_state(self, resume_from):
        """加载训练状态"""
        if not self.state_file.exists():
            self.logger.warning(f"状态文件不存在: {self.state_file}")
            self.current_iteration = resume_from
            return
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.current_iteration = resume_from
            self.iteration_history = state.get('iteration_history', [])
            self.logger.info(f"从迭代 {resume_from} 恢复训练")
            self.logger.info(f"已加载 {len(self.iteration_history)} 条历史记录")
            
            # 恢复不确定性选择器的已知化学空间
            if hasattr(self, 'uncertainty_selector') and self.uncertainty_selector:
                try:
                    self.uncertainty_selector.load_known_space(resume_from)
                    self.logger.info("不确定性选择器状态已恢复")
                except Exception as e:
                    self.logger.warning(f"恢复不确定性选择器状态失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"加载状态失败: {e}")
            self.current_iteration = resume_from
    
    def _cleanup_memory(self):
        """清理内存并监控使用情况"""
        try:
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 清理分子池
            self.current_molecules_pool.clear()
            
            # 强制垃圾回收
            gc.collect()
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_mem_cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                self.logger.debug(f"GPU内存: 已分配={gpu_mem_allocated:.1f}MB, 缓存={gpu_mem_cached:.1f}MB")
            
            mem_after = process.memory_info().rss / 1024 / 1024
            freed = mem_before - mem_after
            self.logger.debug(f"系统内存: {mem_before:.1f}MB → {mem_after:.1f}MB (释放: {freed:.1f}MB)")
            
        except ImportError:
            # 如果psutil未安装，使用基础清理
            self.current_molecules_pool.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.debug("内存已清理（未安装psutil，无法显示详细信息）")
        
    def generate_molecules(self, model, n_samples, iteration):
        """
        使用模型生成分子
        
        Args:
            model: 生成模型
            n_samples: 要生成的分子数量
            iteration: 当前迭代次数
        
        Returns:
            生成的分子列表（RDKit Mol对象）
        """
        self.logger.info("="*60)
        self.logger.info(f"迭代 {iteration}: 生成 {n_samples} 个分子...")
        self.logger.info("="*60)
        
        model.eval()
        
        # 准备参数（已在__init__中验证互斥性）
        pocket_ids = self.args.pocket_ids.split(',') if self.args.pocket_ids else None
        ref_ligand = self.args.ref_ligand
        
        self.logger.debug(f"生成参数: pocket_ids={pocket_ids}, ref_ligand={ref_ligand}")
        
        try:
            molecules = model.generate_ligands(
                pdb_file=self.args.pdbfile,
                n_samples=n_samples,
                pocket_ids=pocket_ids,
                ref_ligand=ref_ligand,
                sanitize=True,
                largest_frag=True,
                relax_iter=0,
                timesteps=self.args.timesteps
            )
        except Exception as e:
            self.logger.error(f"分子生成失败: {e}")
            raise
        
        self.logger.info(f"成功生成 {len(molecules)} 个有效分子")
        
        # 保存生成的分子
        output_file = self.molecules_dir / f"iteration_{iteration}_generated.sdf"
        utils.write_sdf_file(output_file, molecules)
        self.logger.info(f"分子已保存到: {output_file}")
        
        return molecules
    
    def select_best_molecules(self, molecules, n_select, iteration):
        """
        从分子列表中选择最优的n_select个分子
        使用基于不确定性的智能选择策略
        
        Args:
            molecules: 分子列表（RDKit Mol对象）
            n_select: 要选择的分子数量
            iteration: 当前迭代次数
        
        Returns:
            选中的最优分子列表及其评分
        """
        self.logger.info(f"\n从 {len(molecules)} 个分子中选择最优的 {n_select} 个...")
        
        # 评估所有分子
        scores_df = self.evaluator.evaluate_molecules(molecules)
        
        # 保存原始评分
        scores_file = self.logs_dir / f"iteration_{iteration}_scores.csv"
        scores_df.to_csv(scores_file, index=False)
        self.logger.info(f"评分已保存到: {scores_file}")
        
        # 使用不确定性选择策略
        selected_molecules, selected_scores_df = self.uncertainty_selector.select_molecules(
            molecules, scores_df, n_select, iteration
        )
        
        # 保存选中的分子
        output_file = self.molecules_dir / f"iteration_{iteration}_selected.sdf"
        utils.write_sdf_file(output_file, selected_molecules)
        self.logger.info(f"选中的分子已保存到: {output_file}")
        
        # 记录统计信息
        self.logger.info("\n选中分子的详细评分统计:")
        display_cols = ['QED', 'SA', 'LogP', 'Lipinski', '综合得分', 'uncertainty', 'combined_score']
        available_cols = [col for col in display_cols if col in selected_scores_df.columns]
        if 'Docking_Score' in selected_scores_df.columns:
            available_cols.insert(4, 'Docking_Score')
        stats = selected_scores_df[available_cols].describe()
        self.logger.info(f"\n{stats}")
        
        # 存储到内存池（用于后续迭代）
        # 清理旧的池以释放内存
        self.current_molecules_pool.clear()
        self.current_molecules_pool = selected_molecules
        
        return selected_molecules, selected_scores_df
    
    def train_on_selected_molecules(self, molecules, iteration):
        """
        使用选中的分子训练模型（固定底层，只训练上层）
        
        Args:
            molecules: 用于训练的分子列表
            iteration: 当前迭代次数
        
        Returns:
            训练后的模型检查点路径
        """
        self.logger.info("="*60)
        self.logger.info(f"迭代 {iteration}: 准备训练数据并训练模型...")
        self.logger.info("="*60)
        
        # 准备训练数据（将分子转换为npz格式）
        data_file = self.data_dir / f"iteration_{iteration}_train.npz"
        prepare_iterative_training_data(
            molecules=molecules,
            pdb_file=self.args.pdbfile,
            ref_ligand=self.args.ref_ligand,
            pocket_ids=self.args.pocket_ids.split(',') if self.args.pocket_ids else None,
            output_file=data_file,
            logger=self.logger
        )
        
        # 确定使用的检查点
        if iteration == 1:
            checkpoint = self.args.checkpoint
        else:
            prev_checkpoint = self.models_dir / f"iteration_{iteration-1}_checkpoint.ckpt"
            if not prev_checkpoint.exists():
                self.logger.warning(f"上一次迭代的检查点不存在: {prev_checkpoint}")
                self.logger.info("使用初始检查点")
                checkpoint = self.args.checkpoint
            else:
                checkpoint = str(prev_checkpoint)
        
        # 训练模型（固定底层EGNN层，只训练上层）
        checkpoint_path = self.models_dir / f"iteration_{iteration}_checkpoint.ckpt"
        train_with_frozen_layers(
            checkpoint=checkpoint,
            data_file=data_file,
            output_checkpoint=checkpoint_path,
            n_epochs=self.args.train_epochs,
            freeze_bottom_layers=self.args.freeze_layers,
            batch_size=self.args.batch_size,
            lr=self.args.lr,
            logger=self.logger
        )
        
        return checkpoint_path
    
    def run_iteration(self, iteration):
        """
        运行一次完整的迭代
        
        Args:
            iteration: 迭代次数
        """
        self.logger.info("\n" + "#"*70)
        self.logger.info("#"*70)
        self.logger.info(f"开始第 {iteration} 次迭代")
        self.logger.info("#"*70)
        self.logger.info("#"*70 + "\n")
        
        # 加载当前模型
        if iteration == 1:
            current_model = self.initial_model
            # 第一次迭代：生成2000个分子
            n_generate = 2000
        else:
            # 后续迭代：加载上一次训练的模型
            prev_checkpoint = self.models_dir / f"iteration_{iteration-1}_checkpoint.ckpt"
            if not prev_checkpoint.exists():
                self.logger.warning(f"检查点文件不存在: {prev_checkpoint}")
                self.logger.info("使用初始模型")
                current_model = self.initial_model
            else:
                self.logger.info(f"加载模型: {prev_checkpoint}")
                current_model = LigandPocketDDPM.load_from_checkpoint(
                    prev_checkpoint, map_location=self.device
                )
                current_model = current_model.to(self.device)
            # 后续迭代：生成1000个新分子
            n_generate = 1000
        
        # 步骤1: 生成分子
        new_molecules = self.generate_molecules(current_model, n_generate, iteration)
        
        # 清理模型以释放内存（如果不是初始模型）
        if iteration > 1 and current_model is not self.initial_model:
            del current_model
            self._cleanup_memory()
        
        # 步骤2: 合并分子池（使用内存池而不是重新读取SDF）
        if iteration == 1:
            candidate_molecules = new_molecules
        else:
            # 使用内存中的分子池
            if self.current_molecules_pool:
                prev_molecules = self.current_molecules_pool
                self.logger.info(f"从内存池加载 {len(prev_molecules)} 个分子")
            else:
                # 如果内存池为空，从文件读取
                from rdkit import Chem
                prev_selected_file = self.molecules_dir / f"iteration_{iteration-1}_selected.sdf"
                prev_molecules = [mol for mol in Chem.SDMolSupplier(str(prev_selected_file)) if mol is not None]
                self.logger.info(f"从文件加载 {len(prev_molecules)} 个分子: {prev_selected_file}")
            
            candidate_molecules = prev_molecules + new_molecules
            self.logger.info(f"\n合并分子池: {len(prev_molecules)} (上次选中) + {len(new_molecules)} (新生成) = {len(candidate_molecules)} (总计)")
        
        # 步骤3: 选择最优的1000个分子
        selected_molecules, scores_df = self.select_best_molecules(
            candidate_molecules, 
            n_select=1000, 
            iteration=iteration
        )
        
        # 保存候选数量（用于统计）
        n_candidates = len(candidate_molecules)
        
        # 清理候选分子列表
        del candidate_molecules
        self._cleanup_memory()
        
        # 步骤4: 记录本次迭代的统计信息
        iteration_stats = {
            'iteration': iteration,
            'n_generated': n_generate,
            'n_candidates': n_candidates,
            'n_selected': len(selected_molecules),
            'avg_qed': float(scores_df['QED'].mean()),
            'avg_sa': float(scores_df['SA'].mean()),
            'avg_logp': float(scores_df['LogP'].mean()),
            'avg_lipinski': float(scores_df['Lipinski'].mean()),
            'avg_docking_score': float(scores_df['Docking_Score'].mean()) if 'Docking_Score' in scores_df and scores_df['Docking_Score'].notna().any() else None,
            'avg_综合得分': float(scores_df['综合得分'].mean())
        }
        self.iteration_history.append(iteration_stats)
        
        # 步骤5: 用选中的分子训练模型（除了最后一次迭代）
        if iteration < self.args.n_iterations:
            checkpoint_path = self.train_on_selected_molecules(selected_molecules, iteration)
            self.logger.info(f"\n模型检查点已保存到: {checkpoint_path}")
        
        # 保存迭代历史
        history_df = pd.DataFrame(self.iteration_history)
        history_file = self.logs_dir / "iteration_history.csv"
        history_df.to_csv(history_file, index=False)
        self.logger.info(f"\n迭代历史已保存到: {history_file}")
        
        # 保存训练状态
        self._save_state(iteration)
        
    def run(self):
        """运行完整的迭代学习流程"""
        self.logger.info("="*70)
        self.logger.info("开始迭代学习流程")
        self.logger.info("="*70)
        self.logger.info(f"总迭代次数: {self.args.n_iterations}")
        self.logger.info(f"蛋白文件: {self.args.pdbfile}")
        self.logger.info(f"初始模型: {self.args.checkpoint}")
        self.logger.info(f"输出目录: {self.output_dir}")
        if self.args.resume_from:
            self.logger.info(f"从迭代 {self.args.resume_from} 恢复训练")
        self.logger.info("="*70 + "\n")
        
        start_time = datetime.now()
        
        # 确定起始迭代
        start_iteration = self.current_iteration if self.args.resume_from else 1
        
        # 运行所有迭代
        for iteration in range(start_iteration, self.args.n_iterations + 1):
            try:
                self.run_iteration(iteration)
            except Exception as e:
                self.logger.error(f"\n错误：迭代 {iteration} 失败: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                
                # 继续下一次迭代或停止
                if not self.args.continue_on_error:
                    self.logger.info("停止迭代流程")
                    break
                else:
                    self.logger.info("继续下一次迭代...")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        self.logger.info("\n" + "="*70)
        self.logger.info("迭代学习完成！")
        self.logger.info("="*70)
        self.logger.info(f"总用时: {duration}")
        self.logger.info(f"结果保存在: {self.output_dir}")
        self.logger.info("="*70 + "\n")
        
        # 生成最终报告
        self.generate_final_report()
    
    def generate_final_report(self):
        """生成最终报告"""
        report_file = self.output_dir / "final_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("迭代学习最终报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"项目名称: RE-CmeB蛋白分子优化\n")
            f.write(f"总迭代次数: {self.args.n_iterations}\n")
            f.write(f"初始模型: {self.args.checkpoint}\n")
            f.write(f"蛋白文件: {self.args.pdbfile}\n")
            f.write(f"输出目录: {self.output_dir}\n\n")
            
            f.write("迭代历史统计:\n")
            f.write("-" * 70 + "\n")
            
            if self.iteration_history:
                df = pd.DataFrame(self.iteration_history)
                f.write(df.to_string(index=False))
                f.write("\n\n")
                
                # 绘制趋势
                f.write("性能趋势:\n")
                f.write("-" * 70 + "\n")
                f.write(f"QED: {df['avg_qed'].iloc[0]:.3f} -> {df['avg_qed'].iloc[-1]:.3f}\n")
                f.write(f"SA: {df['avg_sa'].iloc[0]:.3f} -> {df['avg_sa'].iloc[-1]:.3f}\n")
                f.write(f"LogP: {df['avg_logp'].iloc[0]:.3f} -> {df['avg_logp'].iloc[-1]:.3f}\n")
                f.write(f"Lipinski: {df['avg_lipinski'].iloc[0]:.3f} -> {df['avg_lipinski'].iloc[-1]:.3f}\n")
                if df['avg_docking_score'].notna().any():
                    f.write(f"Docking Score: {df['avg_docking_score'].iloc[0]:.3f} -> {df['avg_docking_score'].iloc[-1]:.3f}\n")
                f.write(f"综合得分: {df['avg_综合得分'].iloc[0]:.3f} -> {df['avg_综合得分'].iloc[-1]:.3f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("报告生成完成\n")
            f.write("=" * 70 + "\n")
        
        print(f"最终报告已保存到: {report_file}")


def load_config(config_file):
    """从YAML文件加载配置"""
    import yaml
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def config_to_args(config):
    """将配置字典转换为命令行参数格式"""
    args_dict = {}
    
    # 路径配置
    if 'paths' in config:
        args_dict['checkpoint'] = config['paths'].get('checkpoint')
        args_dict['pdbfile'] = config['paths'].get('pdbfile')
        args_dict['output_dir'] = config['paths'].get('output_dir')
    
    # 口袋定义
    if 'pocket' in config:
        args_dict['ref_ligand'] = config['pocket'].get('ref_ligand')
        args_dict['pocket_ids'] = config['pocket'].get('pocket_ids')
        # 如果pocket_ids是列表，转换为逗号分隔的字符串
        if isinstance(args_dict['pocket_ids'], list):
            args_dict['pocket_ids'] = ','.join(args_dict['pocket_ids'])
    
    # 迭代参数
    if 'iteration' in config:
        args_dict['n_iterations'] = config['iteration'].get('n_iterations', 30)
        args_dict['train_epochs'] = config['iteration'].get('train_epochs', 50)
        args_dict['freeze_layers'] = config['iteration'].get('freeze_layers', 3)
    
    # 训练参数
    if 'training' in config:
        args_dict['batch_size'] = config['training'].get('batch_size', 8)
        args_dict['lr'] = config['training'].get('learning_rate', 1e-4)
    
    # 生成参数
    if 'generation' in config:
        args_dict['timesteps'] = config['generation'].get('timesteps')
    
    # 评估参数
    if 'evaluation' in config:
        args_dict['use_docking'] = config['evaluation'].get('use_docking', False)
    
    # 不确定性参数
    if 'uncertainty' in config:
        args_dict['alpha_start'] = config['uncertainty'].get('alpha_start', 0.5)
        args_dict['alpha_end'] = config['uncertainty'].get('alpha_end', 0.03)
    
    # 恢复训练
    if 'resume' in config:
        args_dict['resume_from'] = config['resume'].get('resume_from')
        args_dict['continue_on_error'] = config['resume'].get('continue_on_error', False)
    
    return args_dict


def main():
    parser = argparse.ArgumentParser(description='迭代学习分子生成系统')
    
    # 配置文件和预设参数
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（YAML格式）')
    parser.add_argument('--preset', type=str, default=None,
                        choices=['default', 'quick_test', 'high_quality', 'fast_training'],
                        help='使用预设配置')
    
    # 必需参数（如果使用配置文件则变为可选）
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='初始模型检查点路径')
    parser.add_argument('--pdbfile', type=str, default=None,
                        help='蛋白质PDB文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录路径')
    
    # 口袋定义（二选一）
    parser.add_argument('--pocket_ids', type=str, default=None,
                        help='口袋残基列表，格式: A:1,A:2,A:3')
    parser.add_argument('--ref_ligand', type=str, default=None,
                        help='参考配体（链:残基ID 或 SDF文件路径）')
    
    # 迭代参数
    parser.add_argument('--n_iterations', type=int, default=30,
                        help='迭代次数（默认30）')
    parser.add_argument('--train_epochs', type=int, default=50,
                        help='每次迭代的训练轮数（默认50）')
    parser.add_argument('--freeze_layers', type=int, default=3,
                        help='冻结的EGNN底层数量（默认3，总共5层）')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='训练批次大小（默认8）')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率（默认1e-4）')
    
    # 生成参数
    parser.add_argument('--timesteps', type=int, default=None,
                        help='去噪步数（默认使用训练值）')
    
    # 评估参数
    parser.add_argument('--use_docking', action='store_true',
                        help='是否使用对接打分（需要安装smina）')
    
    # 不确定性选择参数（NEW）
    parser.add_argument('--alpha_start', type=float, default=0.5,
                        help='初始探索权重（0-1），推荐0.4-0.6。默认0.5')
    parser.add_argument('--alpha_end', type=float, default=0.03,
                        help='最终探索权重（0-1），推荐0.01-0.05。默认0.03')
    
    # 恢复训练
    parser.add_argument('--resume_from', type=int, default=None,
                        help='从指定迭代恢复训练（例如: --resume_from 5）')
    
    # 其他
    parser.add_argument('--continue_on_error', action='store_true',
                        help='遇到错误时继续执行')
    
    args = parser.parse_args()
    
    # 处理配置文件和预设
    config_values = {}
    
    # 1. 如果指定了配置文件，加载配置
    if args.config:
        config = load_config(args.config)
        
        # 如果指定了预设，应用预设覆盖
        if args.preset and 'presets' in config and args.preset in config['presets']:
            preset_config = config['presets'][args.preset]
            # 合并预设配置到主配置
            for key, value in preset_config.items():
                if key == 'train_epochs' and 'iteration' in config:
                    config['iteration']['train_epochs'] = value
                elif key == 'freeze_layers' and 'iteration' in config:
                    config['iteration']['freeze_layers'] = value
                elif key == 'batch_size' and 'training' in config:
                    config['training']['batch_size'] = value
                elif key == 'learning_rate' and 'training' in config:
                    config['training']['learning_rate'] = value
                elif key == 'n_iterations' and 'iteration' in config:
                    config['iteration']['n_iterations'] = value
                elif key == 'initial_molecules' and 'generation' in config:
                    config['generation']['initial_molecules'] = value
                elif key == 'subsequent_molecules' and 'generation' in config:
                    config['generation']['subsequent_molecules'] = value
                elif key == 'n_select' and 'generation' in config:
                    config['generation']['n_select'] = value
                elif key == 'use_docking' and 'evaluation' in config:
                    config['evaluation']['use_docking'] = value
        
        config_values = config_to_args(config)
    
    # 2. 如果只指定了预设（没有配置文件），使用默认config.yaml
    elif args.preset:
        default_config_path = Path(__file__).parent / 'config.yaml'
        if default_config_path.exists():
            config = load_config(default_config_path)
            if 'presets' in config and args.preset in config['presets']:
                preset_config = config['presets'][args.preset]
                # 应用预设
                for key, value in preset_config.items():
                    if key == 'train_epochs' and 'iteration' in config:
                        config['iteration']['train_epochs'] = value
                    elif key == 'freeze_layers' and 'iteration' in config:
                        config['iteration']['freeze_layers'] = value
                    elif key == 'batch_size' and 'training' in config:
                        config['training']['batch_size'] = value
                    elif key == 'learning_rate' and 'training' in config:
                        config['training']['learning_rate'] = value
                    elif key == 'n_iterations' and 'iteration' in config:
                        config['iteration']['n_iterations'] = value
                config_values = config_to_args(config)
    
    # 3. 用配置值填充未设置的命令行参数（命令行参数优先级更高）
    for key, value in config_values.items():
        if value is not None and (not hasattr(args, key) or getattr(args, key) is None):
            setattr(args, key, value)
    
    # 验证必需参数
    if not args.checkpoint:
        parser.error("必须指定 --checkpoint 或在配置文件中设置")
    if not args.pdbfile:
        parser.error("必须指定 --pdbfile 或在配置文件中设置")
    if not args.output_dir:
        parser.error("必须指定 --output_dir 或在配置文件中设置")
    
    # 验证口袋定义（在IterativeLearning.__init__中会再次验证）
    if not ((args.pocket_ids is None) ^ (args.ref_ligand is None)):
        parser.error("必须指定且仅指定 --pocket_ids 或 --ref_ligand 之一")
    
    # 设置日志
    logger = setup_logger(args.output_dir, args.resume_from)
    logger.info("="*70)
    logger.info("DiffSBDD 迭代学习系统")
    logger.info("="*70)
    
    try:
        # 运行迭代学习
        learner = IterativeLearning(args, logger)
        learner.run()
        logger.info("训练完成！")
    except KeyboardInterrupt:
        logger.warning("\n训练被用户中断")
        logger.info("训练状态已保存，可使用 --resume_from 参数恢复")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

