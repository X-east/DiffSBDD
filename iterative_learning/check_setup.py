#!/usr/bin/env python3
"""
环境检查脚本
验证所有依赖是否正确安装
"""

import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print(f"    需要Python 3.8或更高版本")
        return False

def check_import(module_name, package_name=None):
    """检查Python包是否可导入"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError:
        print(f"  ✗ {package_name} 未安装")
        return False

def check_packages():
    """检查所有必需的Python包"""
    print("\n检查Python包...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('rdkit', 'RDKit'),
        ('Bio', 'BioPython'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('yaml', 'PyYAML'),
    ]
    
    all_ok = True
    for module, name in packages:
        if not check_import(module, name):
            all_ok = False
    
    return all_ok

def check_cuda():
    """检查CUDA是否可用"""
    print("\n检查CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA可用")
            print(f"    设备数量: {torch.cuda.device_count()}")
            print(f"    当前设备: {torch.cuda.current_device()}")
            print(f"    设备名称: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print(f"  ! CUDA不可用（将使用CPU，速度较慢）")
            return False
    except:
        print(f"  ! 无法检查CUDA状态")
        return False

def check_files():
    """检查关键文件是否存在"""
    print("\n检查文件...")
    
    files = [
        'iterative_generation.py',
        'molecule_evaluator.py',
        'prepare_training_data.py',
        'train_frozen.py',
        'run_example.sh',
    ]
    
    all_ok = True
    for filename in files:
        if Path(filename).exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} 不存在")
            all_ok = False
    
    return all_ok

def check_directories():
    """检查目录结构"""
    print("\n检查目录...")
    
    dirs = {
        'checkpoints': '模型检查点目录',
        'proteins': '蛋白文件目录',
        'results': '结果输出目录',
    }
    
    for dirname, desc in dirs.items():
        if Path(dirname).exists():
            print(f"  ✓ {dirname}/ ({desc})")
        else:
            print(f"  ! {dirname}/ 不存在 ({desc}) - 将在运行时创建")

def check_model():
    """检查预训练模型"""
    print("\n检查预训练模型...")
    
    model_path = Path('checkpoints/crossdocked_fullatom_cond.ckpt')
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"  ✓ 预训练模型已下载 ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ✗ 预训练模型未找到")
        print(f"    请运行: ./setup_environment.sh")
        print(f"    或手动下载: wget -P checkpoints/ https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt")
        return False

def check_smina():
    """检查smina是否安装"""
    print("\n检查对接软件...")
    
    import shutil
    if shutil.which('smina.static'):
        print(f"  ✓ smina已安装（可使用对接打分）")
        return True
    else:
        print(f"  ! smina未安装（无法使用对接打分）")
        print(f"    这是可选的，如需使用请运行: ./setup_environment.sh")
        return False

def check_parent_modules():
    """检查父目录模块"""
    print("\n检查DiffSBDD模块...")
    
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    
    modules = [
        'lightning_modules',
        'dataset',
        'constants',
        'utils',
    ]
    
    all_ok = True
    for module in modules:
        if not check_import(module):
            all_ok = False
    
    return all_ok

def main():
    print("=" * 70)
    print("DiffSBDD 迭代学习系统 - 环境检查")
    print("=" * 70)
    
    # 改变到脚本所在目录
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    checks = []
    
    # 运行所有检查
    checks.append(("Python版本", check_python_version()))
    checks.append(("Python包", check_packages()))
    checks.append(("CUDA", check_cuda()))
    checks.append(("核心文件", check_files()))
    checks.append(("DiffSBDD模块", check_parent_modules()))
    
    check_directories()  # 只是提示，不算失败
    checks.append(("预训练模型", check_model()))
    check_smina()  # 可选，不算失败
    
    # 总结
    print("\n" + "=" * 70)
    print("检查总结")
    print("=" * 70)
    
    for name, result in checks:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s} {status}")
    
    all_passed = all(result for _, result in checks)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ 所有检查通过！可以开始运行迭代学习系统")
        print("\n下一步:")
        print("1. 将RE-CmeB蛋白PDB文件放到 proteins/ 目录")
        print("2. 编辑 run_example.sh 配置参数")
        print("3. 运行: ./run_example.sh")
    else:
        print("✗ 部分检查失败，请先解决上述问题")
        print("\n建议:")
        print("1. 运行: ./setup_environment.sh")
        print("2. 激活环境: conda activate diffsbdd")
        print("3. 重新运行此检查: python check_setup.py")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

