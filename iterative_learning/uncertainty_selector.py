"""
基于不确定性的分子选择策略
整合到DiffSBDD迭代学习系统中

核心思想：
- 前期（迭代1-10）：大胆探索未知化学空间（高alpha）
- 中期（迭代11-20）：平衡探索与利用（中等alpha）
- 后期（迭代21-30）：聚焦优化最优分子（低alpha）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from typing import List, Tuple
import pickle
import json


class UncertaintyBasedSelector:
    """
    基于不确定性的分子选择器
    
    不确定性度量：U(m) = 1 - max_similarity(m, known_space)
    综合得分：S(m) = (1-α)·Q(m) + α·U(m)
    
    其中：
    - Q(m): 质量得分（QED, SA, LogP, Lipinski, Docking等）
    - U(m): 不确定性得分（化学空间探索价值）
    - α: 探索权重（随迭代衰减）
    """
    
    def __init__(self, 
                 output_dir: Path,
                 alpha_start: float = 0.5,
                 alpha_end: float = 0.03,
                 n_iterations: int = 30,
                 fp_radius: int = 2,
                 fp_bits: int = 2048,
                 logger=None):
        """
        Args:
            output_dir: 输出目录
            alpha_start: 初始探索权重（前期探索）
            alpha_end: 最终探索权重（后期优化）
            n_iterations: 总迭代次数
            fp_radius: Morgan指纹半径
            fp_bits: 指纹位数
            logger: 日志对象
        """
        import logging
        self.output_dir = Path(output_dir)
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.n_iterations = n_iterations
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.logger = logger or logging.getLogger(__name__)
        
        # 历史记录
        self.known_fps = []  # 已知分子指纹
        self.known_smiles = []  # 已知分子SMILES
        self.selection_history = []  # 选择历史
        
        # 创建输出目录
        self.uncertainty_dir = self.output_dir / "uncertainty_analysis"
        self.uncertainty_dir.mkdir(exist_ok=True, parents=True)
    
    def compute_alpha(self, iteration: int) -> float:
        """
        计算当前迭代的探索权重（线性衰减）
        
        迭代1:  α ≈ 0.50 (50%探索 + 50%利用) - 大胆探索
        迭代10: α ≈ 0.35 (35%探索 + 65%利用) - 广泛搜索
        迭代20: α ≈ 0.19 (19%探索 + 81%利用) - 聚焦优化
        迭代30: α ≈ 0.03 (3%探索 + 97%利用)  - 精细优化
        """
        if self.n_iterations == 1:
            return self.alpha_start
        
        t = (iteration - 1) / (self.n_iterations - 1)  # 归一化到[0,1]
        alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * t
        return np.clip(alpha, self.alpha_end, self.alpha_start)
    
    def compute_fingerprints(self, molecules: List[Chem.Mol]) -> List:
        """
        计算分子指纹（并行化处理）
        
        Args:
            molecules: 分子列表
            
        Returns:
            指纹列表
        """
        # 对于少量分子，使用串行处理更快
        if len(molecules) < 100:
            return self._compute_fingerprints_serial(molecules)
        
        # 对于大量分子，使用并行处理
        try:
            from multiprocessing import Pool, cpu_count
            import os
            
            # 使用CPU核心数的一半，避免过载
            n_processes = max(1, cpu_count() // 2)
            
            self.logger.debug(f"并行计算指纹: {len(molecules)} 个分子, {n_processes} 进程")
            
            with Pool(n_processes) as pool:
                fingerprints = pool.starmap(
                    self._compute_single_fingerprint,
                    [(mol, self.fp_radius, self.fp_bits) for mol in molecules]
                )
            
            return fingerprints
            
        except Exception as e:
            self.logger.warning(f"并行计算失败，使用串行处理: {e}")
            return self._compute_fingerprints_serial(molecules)
    
    @staticmethod
    def _compute_single_fingerprint(mol, radius, n_bits):
        """
        计算单个分子指纹（静态方法，用于并行处理）
        
        Args:
            mol: 分子对象
            radius: Morgan指纹半径
            n_bits: 指纹位数
            
        Returns:
            指纹对象或None
        """
        if mol is None:
            return None
        try:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        except:
            return None
    
    def _compute_fingerprints_serial(self, molecules: List[Chem.Mol]) -> List:
        """串行计算分子指纹（备用方法）"""
        fingerprints = []
        for mol in molecules:
            if mol is not None:
                try:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.fp_radius, nBits=self.fp_bits
                    )
                    fingerprints.append(fp)
                except Exception as e:
                    self.logger.debug(f"计算指纹失败: {e}")
                    fingerprints.append(None)
            else:
                fingerprints.append(None)
        return fingerprints
    
    def compute_uncertainty_scores(self, 
                                   candidate_fps: List,
                                   known_fps: List) -> np.ndarray:
        """
        计算候选分子相对于已知空间的不确定性
        
        不确定性 = 1 - 最大相似度
        - 1.0: 完全新颖（与已知分子完全不同）
        - 0.5: 中等新颖
        - 0.0: 完全已知（与某个已知分子完全相同）
        """
        if not known_fps:
            # 第一次迭代，所有分子都是新的
            self.logger.info("第一次迭代：所有分子不确定性=1.0（全探索）")
            return np.ones(len(candidate_fps))
        
        uncertainty_scores = []
        
        for i, cand_fp in enumerate(candidate_fps):
            if cand_fp is None:
                uncertainty_scores.append(0.0)
                continue
            
            # 计算与所有已知分子的相似度
            similarities = [
                DataStructs.TanimotoSimilarity(cand_fp, known_fp)
                for known_fp in known_fps
                if known_fp is not None
            ]
            
            if similarities:
                max_similarity = max(similarities)
                uncertainty = 1.0 - max_similarity
            else:
                uncertainty = 1.0
            
            uncertainty_scores.append(uncertainty)
        
        return np.array(uncertainty_scores)
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max归一化到[0, 1]"""
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        min_val = scores.min()
        max_val = scores.max()
        
        if max_val - min_val < 1e-10:
            self.logger.warning("得分无差异，归一化为0.5")
            return np.ones_like(scores) * 0.5
        
        return (scores - min_val) / (max_val - min_val)
    
    def select_molecules(self,
                        molecules: List[Chem.Mol],
                        scores_df: pd.DataFrame,
                        n_select: int,
                        iteration: int) -> Tuple[List[Chem.Mol], pd.DataFrame]:
        """
        基于不确定性选择分子
        
        Returns:
            (selected_molecules, selected_scores_df)
        """
        self.logger.info("="*70)
        self.logger.info(f"【基于不确定性的智能选择】迭代 {iteration}")
        self.logger.info("="*70)
        
        # 1. 计算当前探索权重
        alpha = self.compute_alpha(iteration)
        self.logger.info(f"探索权重 α = {alpha:.3f} ({alpha*100:.1f}%探索 + {(1-alpha)*100:.1f}%利用)")
        
        # 2. 计算分子指纹
        self.logger.info("计算分子指纹...")
        candidate_fps = self.compute_fingerprints(molecules)
        valid_count = sum(1 for fp in candidate_fps if fp is not None)
        self.logger.info(f"有效指纹: {valid_count}/{len(candidate_fps)}")
        
        # 3. 计算不确定性
        self.logger.info("计算不确定性分数...")
        uncertainty_scores = self.compute_uncertainty_scores(
            candidate_fps, self.known_fps
        )
        
        # 4. 获取质量得分
        quality_scores = scores_df['综合得分'].values
        
        # 5. 归一化
        quality_norm = self.normalize_scores(quality_scores)
        uncertainty_norm = self.normalize_scores(uncertainty_scores)
        
        # 6. 计算综合得分
        combined_scores = (1 - alpha) * quality_norm + alpha * uncertainty_norm
        
        # 7. 处理无效指纹（给予惩罚）
        for i, fp in enumerate(candidate_fps):
            if fp is None:
                combined_scores[i] = 0.0
        
        # 8. 选择top-k
        if len(combined_scores) <= n_select:
            self.logger.warning(f"候选数({len(combined_scores)}) <= 选择数({n_select})，全部选择")
            selected_indices = list(range(len(combined_scores)))
        else:
            selected_indices = np.argsort(combined_scores)[-n_select:]
        
        selected_molecules = [molecules[i] for i in selected_indices]
        selected_scores_df = scores_df.iloc[selected_indices].copy()
        
        # 9. 添加额外信息到DataFrame
        selected_scores_df['uncertainty'] = uncertainty_scores[selected_indices]
        selected_scores_df['combined_score'] = combined_scores[selected_indices]
        
        # 10. 计算统计信息
        stats = self._compute_statistics(
            quality_scores, uncertainty_scores, combined_scores,
            selected_indices, alpha, iteration
        )
        
        # 11. 更新已知空间
        self._update_known_space(selected_molecules, selected_indices, candidate_fps)
        
        # 12. 保存记录
        self._save_records(iteration, stats, selected_scores_df)
        
        # 13. 打印详细统计
        self._print_statistics(stats)
        
        return selected_molecules, selected_scores_df
    
    def _compute_statistics(self, quality_scores, uncertainty_scores, 
                          combined_scores, selected_indices, alpha, iteration):
        """计算详细统计信息"""
        
        # 纯贪心对比
        greedy_indices = np.argsort(quality_scores)[-len(selected_indices):]
        
        stats = {
            'iteration': iteration,
            'alpha': float(alpha),
            'n_candidates': len(quality_scores),
            'n_selected': len(selected_indices),
            
            # 质量得分
            'quality_mean_all': float(np.mean(quality_scores)),
            'quality_mean_selected': float(np.mean(quality_scores[selected_indices])),
            'quality_mean_greedy': float(np.mean(quality_scores[greedy_indices])),
            'quality_max_selected': float(np.max(quality_scores[selected_indices])),
            'quality_min_selected': float(np.min(quality_scores[selected_indices])),
            
            # 不确定性
            'uncertainty_mean_all': float(np.mean(uncertainty_scores)),
            'uncertainty_mean_selected': float(np.mean(uncertainty_scores[selected_indices])),
            'uncertainty_max_selected': float(np.max(uncertainty_scores[selected_indices])),
            'uncertainty_min_selected': float(np.min(uncertainty_scores[selected_indices])),
            
            # 综合得分
            'combined_mean_selected': float(np.mean(combined_scores[selected_indices])),
            
            # 已知空间
            'known_space_size': len(self.known_fps)
        }
        
        # 质量差距（负值表示我们更好）
        stats['quality_gap'] = stats['quality_mean_greedy'] - stats['quality_mean_selected']
        
        return stats
    
    def _update_known_space(self, selected_molecules, selected_indices, candidate_fps):
        """更新已知化学空间"""
        new_fps = [candidate_fps[i] for i in selected_indices if candidate_fps[i] is not None]
        new_smiles = [Chem.MolToSmiles(mol) for mol in selected_molecules if mol is not None]
        
        self.known_fps.extend(new_fps)
        self.known_smiles.extend(new_smiles)
        
        self.logger.info(f"已知空间更新: {len(self.known_fps)} 个分子")
    
    def _save_records(self, iteration, stats, selected_scores_df):
        """保存选择记录"""
        # 保存统计信息
        self.selection_history.append(stats)
        
        stats_file = self.uncertainty_dir / f"iteration_{iteration}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 保存详细得分
        scores_file = self.uncertainty_dir / f"iteration_{iteration}_detailed_scores.csv"
        selected_scores_df.to_csv(scores_file, index=False)
        
        # 保存历史汇总
        history_file = self.uncertainty_dir / "selection_history.csv"
        history_df = pd.DataFrame(self.selection_history)
        history_df.to_csv(history_file, index=False)
        
        self.logger.info(f"记录已保存: {self.uncertainty_dir}")
    
    def _print_statistics(self, stats):
        """打印统计信息"""
        self.logger.info("\n" + "-"*70)
        self.logger.info("【选择统计】")
        self.logger.info("-"*70)
        self.logger.info(f"候选数量: {stats['n_candidates']} → 选择: {stats['n_selected']}")
        self.logger.info(f"")
        self.logger.info(f"质量得分:")
        self.logger.info(f"  全局平均:   {stats['quality_mean_all']:.4f}")
        self.logger.info(f"  选中平均:   {stats['quality_mean_selected']:.4f} [{stats['quality_min_selected']:.4f}, {stats['quality_max_selected']:.4f}]")
        self.logger.info(f"  纯贪心平均: {stats['quality_mean_greedy']:.4f}")
        self.logger.info(f"  质量差距:   {stats['quality_gap']:.4f} {'✓更好' if stats['quality_gap'] < 0 else '(略低但探索更多)'}")
        self.logger.info(f"")
        self.logger.info(f"不确定性（化学空间探索）:")
        self.logger.info(f"  全局平均:   {stats['uncertainty_mean_all']:.4f}")
        self.logger.info(f"  选中平均:   {stats['uncertainty_mean_selected']:.4f} [{stats['uncertainty_min_selected']:.4f}, {stats['uncertainty_max_selected']:.4f}]")
        self.logger.info(f"")
        self.logger.info(f"已知化学空间: {stats['known_space_size']} 个分子")
        self.logger.info("-"*70 + "\n")
    
    def load_known_space(self, iteration: int):
        """
        加载已知空间（用于恢复训练）
        从之前的迭代中重建已知分子空间
        """
        # 1. 加载选择历史
        history_file = self.uncertainty_dir / "selection_history.csv"
        if history_file.exists():
            history_df = pd.read_csv(history_file)
            self.selection_history = history_df.to_dict('records')
            self.logger.info(f"加载选择历史: {len(self.selection_history)} 条记录")
        
        # 2. 从之前迭代的选中分子重建已知空间
        molecules_dir = self.output_dir / "molecules"
        if not molecules_dir.exists():
            self.logger.warning("分子目录不存在，无法恢复已知空间")
            return
        
        self.known_fps = []
        self.known_smiles = []
        
        # 读取从迭代1到当前迭代-1的所有选中分子
        for i in range(1, iteration):
            selected_file = molecules_dir / f"iteration_{i}_selected.sdf"
            if selected_file.exists():
                try:
                    suppl = Chem.SDMolSupplier(str(selected_file))
                    mols = [mol for mol in suppl if mol is not None]
                    
                    # 计算指纹和SMILES
                    for mol in mols:
                        try:
                            fp = AllChem.GetMorganFingerprintAsBitVect(
                                mol, self.fp_radius, nBits=self.fp_bits
                            )
                            smiles = Chem.MolToSmiles(mol)
                            self.known_fps.append(fp)
                            self.known_smiles.append(smiles)
                        except Exception as e:
                            self.logger.debug(f"处理分子失败: {e}")
                            continue
                    
                    self.logger.debug(f"从迭代 {i} 加载了 {len(mols)} 个分子")
                    
                except Exception as e:
                    self.logger.warning(f"读取迭代 {i} 的分子失败: {e}")
                    continue
        
        self.logger.info(f"已知化学空间已恢复: {len(self.known_fps)} 个分子")

