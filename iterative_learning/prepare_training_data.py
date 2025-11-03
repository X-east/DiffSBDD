"""
训练数据准备模块
将生成的分子转换为DiffSBDD训练所需的npz格式
"""

import sys
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
import logging

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import dataset_params
import utils


class DataPreparer:
    """数据准备器"""
    
    def __init__(self, dataset='crossdock', pocket_representation='full-atom'):
        """
        初始化数据准备器
        
        Args:
            dataset: 数据集名称（crossdock或moad）
            pocket_representation: 口袋表示方式（CA或full-atom）
        """
        self.dataset_info = dataset_params[dataset]
        self.pocket_representation = pocket_representation
        
        self.atom_encoder = self.dataset_info['atom_encoder']
        self.atom_decoder = self.dataset_info['atom_decoder']
        
        if pocket_representation == 'CA':
            self.aa_encoder = self.dataset_info['aa_encoder']
        
    def get_pocket_from_ligand(self, pdb_model, ref_ligand, dist_cutoff=8.0):
        """
        从配体周围提取口袋残基
        
        Args:
            pdb_model: BioPython PDB模型
            ref_ligand: 参考配体（链:残基ID 或 SDF文件路径）
            dist_cutoff: 距离阈值（埃）
        
        Returns:
            口袋残基列表
        """
        if ref_ligand.endswith(".sdf"):
            # 配体为SDF文件
            rdmol = Chem.SDMolSupplier(str(ref_ligand))[0]
            ligand_coords = torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
            resi = None
        else:
            # 配体在PDB中，格式：链:残基ID
            chain, resi = ref_ligand.split(':')
            ligand = utils.get_residue_with_resi(pdb_model[chain], int(resi))
            ligand_coords = torch.from_numpy(
                np.array([a.get_coord() for a in ligand.get_atoms()]))
        
        pocket_residues = []
        for residue in pdb_model.get_residues():
            # 跳过配体本身（仅当配体在PDB中时）
            if resi is not None and residue.id[1] == resi:
                continue
            
            res_coords = torch.from_numpy(
                np.array([a.get_coord() for a in residue.get_atoms()]))
            
            if is_aa(residue.get_resname(), standard=True) \
                    and torch.cdist(res_coords, ligand_coords).min() < dist_cutoff:
                pocket_residues.append(residue)
        
        return pocket_residues
    
    def get_pocket_from_ids(self, pdb_model, pocket_ids):
        """
        从残基ID列表获取口袋残基
        
        Args:
            pdb_model: BioPython PDB模型
            pocket_ids: 残基ID列表，格式["A:1", "A:2", ...]
        
        Returns:
            口袋残基列表
        """
        residues = []
        for res_id in pocket_ids:
            chain, resi = res_id.split(':')
            residue = utils.get_residue_with_resi(pdb_model[chain], int(resi))
            residues.append(residue)
        return residues
    
    def encode_pocket(self, pocket_residues):
        """
        编码口袋信息
        
        Args:
            pocket_residues: 口袋残基列表
        
        Returns:
            pocket_coords: 口袋坐标
            pocket_one_hot: 口袋类型的one-hot编码
        """
        if self.pocket_representation == 'CA':
            # 仅使用Cα原子
            pocket_coords = np.array(
                [res['CA'].get_coord() for res in pocket_residues])
            pocket_types = np.array([
                self.aa_encoder[three_to_one(res.get_resname())]
                for res in pocket_residues
            ])
            pocket_one_hot = np.eye(len(self.aa_encoder))[pocket_types]
        else:
            # 使用全原子表示
            pocket_atoms = [
                a for res in pocket_residues
                for a in res.get_atoms()
                if (a.element.capitalize() in self.atom_encoder or a.element != 'H')
            ]
            pocket_coords = np.array([a.get_coord() for a in pocket_atoms])
            pocket_types = np.array([
                self.atom_encoder[a.element.capitalize()]
                for a in pocket_atoms
            ])
            pocket_one_hot = np.eye(len(self.atom_encoder))[pocket_types]
        
        return pocket_coords.astype(np.float32), pocket_one_hot.astype(np.float32)
    
    @staticmethod
    def ensure_3d_conformation(mol):
        """
        确保分子有3D构象，如果没有则生成
        
        Args:
            mol: RDKit分子对象
        
        Returns:
            带3D构象的分子
        """
        if mol is None:
            return None
        
        # 检查是否已有3D构象
        try:
            mol.GetConformer()
            return mol  # 已有构象
        except ValueError:
            # 没有构象，需要生成
            pass
        
        # 添加氢原子（可选，有助于生成更好的构象）
        mol_with_h = Chem.AddHs(mol)
        
        # 生成3D构象
        try:
            # 使用ETKDG方法生成3D构象
            AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
            # 使用UFF力场优化
            AllChem.UFFOptimizeMolecule(mol_with_h, maxIters=200)
            # 移除氢原子（保持与原始分子一致）
            mol = Chem.RemoveHs(mol_with_h)
        except Exception as e:
            logging.warning(f"生成3D构象失败: {e}，尝试不添加氢原子")
            try:
                # 确保fallback路径也使用相同的随机种子以保证可重现性
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception as e2:
                logging.error(f"生成3D构象失败: {e2}")
                raise ValueError(f"无法生成3D构象: {e2}")
        
        return mol
    
    def encode_ligand(self, mol):
        """
        编码配体信息
        
        Args:
            mol: RDKit分子对象
        
        Returns:
            lig_coords: 配体坐标
            lig_one_hot: 配体原子类型的one-hot编码
        """
        # 确保分子有3D构象
        mol = self.ensure_3d_conformation(mol)
        
        # 获取配体坐标
        try:
            conf = mol.GetConformer()
            lig_coords = conf.GetPositions().astype(np.float32)
        except Exception as e:
            raise ValueError(f"无法获取分子构象: {e}")
        
        # 获取原子类型
        atom_types = []
        for atom in mol.GetAtoms():
            atom_symbol = atom.GetSymbol()
            if atom_symbol in self.atom_encoder:
                atom_types.append(self.atom_encoder[atom_symbol])
            else:
                # 未知原子类型，使用默认值（通常是C）
                logging.warning(f"未知原子类型 {atom_symbol}，使用C代替")
                atom_types.append(self.atom_encoder['C'])
        
        atom_types = np.array(atom_types)
        lig_one_hot = np.eye(len(self.atom_encoder))[atom_types].astype(np.float32)
        
        return lig_coords, lig_one_hot
    
    def center_complex(self, lig_coords, pocket_coords):
        """
        将配体-口袋复合物居中
        
        Args:
            lig_coords: 配体坐标
            pocket_coords: 口袋坐标
        
        Returns:
            居中后的配体和口袋坐标
        """
        # 计算质心
        all_coords = np.vstack([lig_coords, pocket_coords])
        center = all_coords.mean(axis=0)
        
        # 居中
        lig_coords_centered = lig_coords - center
        pocket_coords_centered = pocket_coords - center
        
        return lig_coords_centered, pocket_coords_centered
    
    def prepare_single_complex(self, mol, pocket_residues):
        """
        准备单个配体-口袋复合物的数据
        
        Args:
            mol: RDKit分子对象
            pocket_residues: 口袋残基列表
        
        Returns:
            数据字典
        """
        # 编码配体
        lig_coords, lig_one_hot = self.encode_ligand(mol)
        
        # 编码口袋
        pocket_coords, pocket_one_hot = self.encode_pocket(pocket_residues)
        
        # 居中
        lig_coords, pocket_coords = self.center_complex(lig_coords, pocket_coords)
        
        return {
            'lig_coords': lig_coords,
            'lig_one_hot': lig_one_hot,
            'pocket_coords': pocket_coords,
            'pocket_one_hot': pocket_one_hot
        }


def prepare_iterative_training_data(molecules, pdb_file, ref_ligand=None, 
                                     pocket_ids=None, output_file=None,
                                     dataset='crossdock', pocket_representation='full-atom',
                                     logger=None):
    """
    准备迭代训练数据
    
    Args:
        molecules: RDKit分子对象列表
        pdb_file: 蛋白PDB文件路径
        ref_ligand: 参考配体（链:残基ID 或 SDF文件）
        pocket_ids: 口袋残基ID列表
        output_file: 输出npz文件路径
        dataset: 数据集名称
        pocket_representation: 口袋表示方式
        logger: 日志对象
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("\n准备训练数据...")
    logger.info(f"分子数量: {len(molecules)}")
    logger.info(f"蛋白文件: {pdb_file}")
    logger.debug(f"数据集: {dataset}, 口袋表示: {pocket_representation}")
    
    # 加载PDB文件
    pdb_parser = PDBParser(QUIET=True)
    pdb_struct = pdb_parser.get_structure('', pdb_file)[0]
    
    # 创建数据准备器
    preparer = DataPreparer(dataset, pocket_representation)
    
    # 获取口袋残基
    if ref_ligand is not None:
        pocket_residues = preparer.get_pocket_from_ligand(pdb_struct, ref_ligand)
        logger.info(f"使用参考配体定义口袋: {ref_ligand}")
    elif pocket_ids is not None:
        pocket_residues = preparer.get_pocket_from_ids(pdb_struct, pocket_ids)
        logger.info(f"使用残基列表定义口袋: {len(pocket_ids)} 个残基")
    else:
        raise ValueError("必须指定ref_ligand或pocket_ids")
    
    logger.info(f"口袋残基数量: {len(pocket_residues)}")
    
    # 准备所有数据
    all_lig_coords = []
    all_lig_one_hot = []
    all_lig_mask = []
    all_pocket_coords = []
    all_pocket_one_hot = []
    all_pocket_mask = []
    names = []
    receptors = []
    
    failed_count = 0
    for i, mol in enumerate(molecules):
        if mol is None:
            failed_count += 1
            continue
        
        try:
            data = preparer.prepare_single_complex(mol, pocket_residues)
            
            all_lig_coords.append(data['lig_coords'])
            all_lig_one_hot.append(data['lig_one_hot'])
            all_lig_mask.append(np.ones(len(data['lig_coords']), dtype=np.int32) * len(names))
            
            all_pocket_coords.append(data['pocket_coords'])
            all_pocket_one_hot.append(data['pocket_one_hot'])
            all_pocket_mask.append(np.ones(len(data['pocket_coords']), dtype=np.int32) * len(names))
            
            names.append(f"mol_{i}")
            receptors.append(Path(pdb_file).stem)
            
        except Exception as e:
            logger.warning(f"处理分子 {i} 失败: {str(e)}")
            failed_count += 1
            continue
    
    # 合并所有数据
    logger.info(f"\n成功处理 {len(names)} 个分子，失败 {failed_count} 个")
    
    if len(names) == 0:
        raise ValueError("没有成功处理的分子，无法生成训练数据")
    
    data_dict = {
        'lig_coords': np.vstack(all_lig_coords).astype(np.float32),
        'lig_one_hot': np.vstack(all_lig_one_hot).astype(np.float32),
        'lig_mask': np.concatenate(all_lig_mask).astype(np.int32),
        'pocket_coords': np.vstack(all_pocket_coords).astype(np.float32),
        'pocket_one_hot': np.vstack(all_pocket_one_hot).astype(np.float32),
        'pocket_mask': np.concatenate(all_pocket_mask).astype(np.int32),
        'names': np.array(names),
        'receptors': np.array(receptors)
    }
    
    # 保存数据
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **data_dict)
        logger.info(f"训练数据已保存到: {output_file}")
        
        # 记录数据形状信息
        logger.debug(f"数据形状 - lig_coords: {data_dict['lig_coords'].shape}")
        logger.debug(f"数据形状 - pocket_coords: {data_dict['pocket_coords'].shape}")
    
    return data_dict


if __name__ == "__main__":
    # 测试代码
    from rdkit import Chem
    
    # 创建测试分子
    smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen
    mol = Chem.MolFromSmiles(smiles)
    
    # 添加3D坐标
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    
    print("测试数据准备器...")
    print(f"分子SMILES: {smiles}")
    print(f"原子数: {mol.GetNumAtoms()}")

