"""
测试冻结策略修复
验证 train_frozen.py 的修复是否正确
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightning_modules import LigandPocketDDPM
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_model_structure():
    """测试模型结构"""
    print("\n" + "="*70)
    print("测试1: 验证模型结构")
    print("="*70)
    
    checkpoint = "../checkpoints/crossdocked_fullatom_cond.ckpt"
    
    try:
        model = LigandPocketDDPM.load_from_checkpoint(
            checkpoint,
            map_location='cpu'
        )
        print("✓ 模型加载成功")
        
        # 验证结构
        assert hasattr(model, 'ddpm'), "模型缺少 ddpm 属性"
        assert hasattr(model.ddpm, 'dynamics'), "模型缺少 dynamics 属性"
        assert hasattr(model.ddpm.dynamics, 'egnn'), "模型缺少 egnn 属性"
        
        print("✓ 模型结构正确")
        
        # 检查EGNN层数
        egnn = model.ddpm.dynamics.egnn
        n_layers = egnn.n_layers
        print(f"✓ EGNN层数: {n_layers}")
        
        # 检查每一层
        print("\nEGNN层结构:")
        for i in range(n_layers):
            block_name = f"e_block_{i}"
            if hasattr(egnn, '_modules') and block_name in egnn._modules:
                block = egnn._modules[block_name]
                num_params = sum(p.numel() for p in block.parameters())
                print(f"  ✓ {block_name}: 存在, {num_params:,} 参数")
            else:
                print(f"  ✗ {block_name}: 不存在")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_freeze_function():
    """测试冻结函数"""
    print("\n" + "="*70)
    print("测试2: 验证冻结函数")
    print("="*70)
    
    checkpoint = "../checkpoints/crossdocked_fullatom_cond.ckpt"
    
    try:
        model = LigandPocketDDPM.load_from_checkpoint(
            checkpoint,
            map_location='cpu'
        )
        
        # 导入冻结函数
        from train_frozen import freeze_model_layers
        
        # 测试默认冻结（3层）
        print("\n测试冻结策略（默认：冻结前3层）")
        model = freeze_model_layers(model, freeze_bottom_layers=3, logger=logger)
        
        # 验证冻结状态
        egnn = model.ddpm.dynamics.egnn
        n_layers = egnn.n_layers
        
        print(f"\n验证冻结状态:")
        all_correct = True
        for i in range(n_layers):
            block_name = f"e_block_{i}"
            if block_name in egnn._modules:
                block = egnn._modules[block_name]
                
                # 检查第一个参数的状态
                first_param = next(block.parameters())
                is_frozen = not first_param.requires_grad
                expected_frozen = (i < 3)
                
                status = "冻结" if is_frozen else "可训练"
                expected_status = "冻结" if expected_frozen else "可训练"
                
                if is_frozen == expected_frozen:
                    print(f"  ✓ {block_name}: {status} (正确)")
                else:
                    print(f"  ✗ {block_name}: {status} (应为 {expected_status})")
                    all_correct = False
        
        if all_correct:
            print("\n✓ 冻结策略正确！")
        else:
            print("\n✗ 冻结策略有误！")
        
        return all_correct
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "#"*70)
    print("# 冻结策略修复验证测试")
    print("#"*70)
    
    test1_pass = test_model_structure()
    test2_pass = test_freeze_function()
    
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    print(f"  测试1 (模型结构): {'✓ 通过' if test1_pass else '✗ 失败'}")
    print(f"  测试2 (冻结函数): {'✓ 通过' if test2_pass else '✗ 失败'}")
    print("="*70)
    
    if test1_pass and test2_pass:
        print("\n🎉 所有测试通过！冻结策略修复成功！")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查代码。")
        return 1


if __name__ == "__main__":
    exit(main())

