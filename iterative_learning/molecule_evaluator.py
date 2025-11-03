"""
分子评估器模块
用于评估生成的分子并根据多个标准进行打分
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
import tempfile

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.SA_Score.sascorer import calculateScore
from analysis.docking import smina_score
import utils


class MoleculeEvaluator:
    """
    分子评估器
    
    评估标准：
    1. QED (Quantitative Estimate of Drug-likeness) - 类药性
    2. SA (Synthetic Accessibility) - 合成可及性
    3. LogP - 亲脂性
    4. Lipinski规则 - 类药五规则
    5. Docking Score - 对接打分（可选）
    """
    
    def __init__(self, pdb_file=None, use_docking=False, logger=None):
        """
        初始化评估器
        
        Args:
            pdb_file: 蛋白PDB文件路径（对接打分需要）
            use_docking: 是否使用对接打分
            logger: 日志对象
        """
        import logging
        self.pdb_file = pdb_file
        self.use_docking = use_docking
        self.logger = logger or logging.getLogger(__name__)
        
        # 评分权重（可根据需要调整）
        self.weights = {
            'QED': 0.25,        # 类药性
            'SA': 0.25,         # 合成可及性
            'LogP': 0.15,       # 亲脂性
            'Lipinski': 0.15,   # Lipinski规则
            'Docking': 0.20     # 对接打分
        }
        
        if not use_docking:
            # 如果不使用对接打分，重新归一化权重（创建新字典避免修改原始值）
            total = sum([v for k, v in self.weights.items() if k != 'Docking'])
            self.weights = {
                k: (v / total if k != 'Docking' else 0.0)
                for k, v in self.weights.items()
            }
            self.logger.info("未启用对接打分，权重已重新分配")
    
    @staticmethod
    def calculate_qed(mol):
        """计算QED（类药性）得分，范围[0,1]，越大越好"""
        try:
            return QED.qed(mol)
        except:
            return 0.0
    
    @staticmethod
    def calculate_sa(mol):
        """
        计算合成可及性得分
        原始SA范围[1,10]，越小越容易合成
        这里转换为[0,1]范围，越大越好
        """
        try:
            sa = calculateScore(mol)
            # 转换：(10 - sa) / 9，使得越容易合成得分越高
            return round((10 - sa) / 9, 3)
        except:
            return 0.0
    
    @staticmethod
    def calculate_logp(mol):
        """
        计算LogP（亲脂性）
        理想范围通常在[-2, 5]之间
        """
        try:
            return Crippen.MolLogP(mol)
        except:
            return 0.0
    
    @staticmethod
    def normalize_logp(logp):
        """
        将LogP归一化到[0,1]范围
        理想范围[-2, 5]，在这个范围内得分较高
        """
        if -2 <= logp <= 5:
            # 在理想范围内，得分为1.0
            return 1.0
        elif logp < -2:
            # 太亲水，线性惩罚
            return max(0, 1.0 + (logp + 2) * 0.1)
        else:
            # 太亲脂，线性惩罚
            return max(0, 1.0 - (logp - 5) * 0.1)
    
    @staticmethod
    def calculate_lipinski(mol):
        """
        计算Lipinski类药五规则得分
        返回满足的规则数量[0-5]
        """
        try:
            rule_1 = Descriptors.ExactMolWt(mol) < 500
            rule_2 = Lipinski.NumHDonors(mol) <= 5
            rule_3 = Lipinski.NumHAcceptors(mol) <= 10
            logp = Crippen.MolLogP(mol)
            rule_4 = (-2 <= logp <= 5)
            rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
            
            return sum([int(r) for r in [rule_1, rule_2, rule_3, rule_4, rule_5]])
        except:
            return 0
    
    @staticmethod
    def normalize_lipinski(lipinski_score):
        """将Lipinski得分归一化到[0,1]"""
        return lipinski_score / 5.0
    
    def calculate_docking_score(self, mols):
        """
        计算对接打分
        使用smina进行对接打分
        
        Args:
            mols: 分子列表
        
        Returns:
            对接分数列表（负数，越负越好）
        """
        if not self.use_docking or self.pdb_file is None:
            return [None] * len(mols)
        
        try:
            scores = smina_score(mols, self.pdb_file)
            return scores
        except Exception as e:
            print(f"对接打分失败: {str(e)}")
            return [None] * len(mols)
    
    @staticmethod
    def normalize_docking_score(score):
        """
        归一化对接分数到[0,1]
        对接分数通常在[-15, 0]范围，越负越好
        """
        if score is None or np.isnan(score):
            return 0.0
        
        # 转换为正值，并归一化
        # 假设-12以下为优秀，0以上为很差
        if score <= -12:
            return 1.0
        elif score >= 0:
            return 0.0
        else:
            return (12 + score) / 12.0
    
    def calculate_综合得分(self, qed, sa, logp, lipinski, docking_score):
        """
        计算综合得分
        
        Args:
            qed: QED得分
            sa: SA得分
            logp: LogP值
            lipinski: Lipinski得分
            docking_score: 对接分数
        
        Returns:
            综合得分[0,1]
        """
        # 归一化各项得分
        qed_norm = qed
        sa_norm = sa
        logp_norm = self.normalize_logp(logp)
        lipinski_norm = self.normalize_lipinski(lipinski)
        docking_norm = self.normalize_docking_score(docking_score)
        
        # 加权求和
        score = (
            self.weights['QED'] * qed_norm +
            self.weights['SA'] * sa_norm +
            self.weights['LogP'] * logp_norm +
            self.weights['Lipinski'] * lipinski_norm
        )
        
        if self.use_docking:
            score += self.weights['Docking'] * docking_norm
        
        return round(score, 4)
    
    def evaluate_molecules(self, molecules):
        """
        评估分子列表
        
        Args:
            molecules: RDKit分子对象列表
        
        Returns:
            包含所有评分的DataFrame
        """
        self.logger.info(f"\n评估 {len(molecules)} 个分子...")
        
        results = []
        
        # 计算基本性质
        self.logger.info("计算分子性质...")
        for i, mol in enumerate(molecules):
            if mol is None:
                continue
            
            try:
                Chem.SanitizeMol(mol)
                
                qed = self.calculate_qed(mol)
                sa = self.calculate_sa(mol)
                logp = self.calculate_logp(mol)
                lipinski = self.calculate_lipinski(mol)
                
                results.append({
                    'mol_id': i,
                    'QED': qed,
                    'SA': sa,
                    'LogP': logp,
                    'Lipinski': lipinski,
                    'mol': mol
                })
            except Exception as e:
                self.logger.warning(f"分子 {i} 评估失败: {str(e)}")
                continue
        
        df = pd.DataFrame(results)
        
        # 计算对接分数
        if self.use_docking and len(results) > 0:
            self.logger.info("计算对接分数...")
            valid_mols = df['mol'].tolist()
            docking_scores = self.calculate_docking_score(valid_mols)
            df['Docking_Score'] = docking_scores
        else:
            df['Docking_Score'] = None
        
        # 计算综合得分
        self.logger.info("计算综合得分...")
        df['综合得分'] = df.apply(
            lambda row: self.calculate_综合得分(
                row['QED'], row['SA'], row['LogP'], 
                row['Lipinski'], row['Docking_Score']
            ),
            axis=1
        )
        
        # 移除mol列（不需要保存到CSV）
        df = df.drop('mol', axis=1)
        
        self.logger.info("\n评估完成！")
        self.logger.info(f"平均 QED: {df['QED'].mean():.3f}")
        self.logger.info(f"平均 SA: {df['SA'].mean():.3f}")
        self.logger.info(f"平均 LogP: {df['LogP'].mean():.3f}")
        self.logger.info(f"平均 Lipinski: {df['Lipinski'].mean():.3f}")
        if self.use_docking and df['Docking_Score'].notna().any():
            self.logger.info(f"平均 Docking Score: {df['Docking_Score'].mean():.3f}")
        self.logger.info(f"平均综合得分: {df['综合得分'].mean():.3f}")
        
        return df


if __name__ == "__main__":
    # 测试代码
    from rdkit import Chem
    
    # 创建一些测试分子
    smiles_list = [
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "CC(=O)Oc1ccccc1C(=O)O",       # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
    ]
    
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    
    evaluator = MoleculeEvaluator(use_docking=False)
    scores = evaluator.evaluate_molecules(mols)
    
    print("\n测试结果:")
    print(scores)

