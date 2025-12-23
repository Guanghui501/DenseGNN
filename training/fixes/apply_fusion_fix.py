#!/usr/bin/env python3
"""
应用融合权重修复脚本

基于诊断结果 α=0.86，修复GatedFusion的初始化。

用法:
    python apply_fusion_fix.py
    python apply_fusion_fix.py --restore  # 恢复原版本
"""

import os
import shutil
from datetime import datetime
import argparse


def backup_original():
    """备份原始文件"""
    original = 'kgcnn/literature/DenseGNN/_multimodal_fusion.py'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = f'kgcnn/literature/DenseGNN/_multimodal_fusion_backup_{timestamp}.py'

    if not os.path.exists(original):
        print(f"❌ 错误: 找不到原文件 {original}")
        return None

    shutil.copy2(original, backup)
    print(f"✓ 已备份到: {backup}")
    return backup


def apply_fix():
    """应用修复"""
    fixed_version = 'kgcnn/literature/DenseGNN/_multimodal_fusion_fixed.py'
    target = 'kgcnn/literature/DenseGNN/_multimodal_fusion.py'

    if not os.path.exists(fixed_version):
        print(f"❌ 错误: 找不到修复版本 {fixed_version}")
        print("   请确保 _multimodal_fusion_fixed.py 存在")
        return False

    # 备份
    backup_file = backup_original()
    if backup_file is None:
        return False

    # 应用修复
    shutil.copy2(fixed_version, target)
    print(f"✓ 已应用修复到: {target}")

    return True


def find_latest_backup():
    """查找最新的备份文件"""
    backup_dir = 'kgcnn/literature/DenseGNN/'
    backups = [f for f in os.listdir(backup_dir)
               if f.startswith('_multimodal_fusion_backup_') and f.endswith('.py')]

    if not backups:
        return None

    backups.sort(reverse=True)
    return os.path.join(backup_dir, backups[0])


def restore_original():
    """恢复原版本"""
    latest_backup = find_latest_backup()

    if latest_backup is None:
        print("❌ 错误: 找不到备份文件")
        return False

    target = 'kgcnn/literature/DenseGNN/_multimodal_fusion.py'
    shutil.copy2(latest_backup, target)
    print(f"✓ 已从备份恢复: {latest_backup}")
    return True


def print_instructions():
    """打印使用说明"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║           融合权重修复已成功应用!                                  ║
╚═══════════════════════════════════════════════════════════════════╝

修改内容:
──────────────────────────────────────────────────────────────────
✓ GatedFusion gate初始化为 0.86 (图86%, 文本14%)
✓ 简化为单一可学习的gate权重
✓ 新增 ResidualFusion 类作为备选方案

下一步操作:
──────────────────────────────────────────────────────────────────
1. 重新训练模型:
   python your_training_script.py

2. 监控训练过程中的gate值:
   在训练脚本中添加:

   if epoch % 10 == 0:
       gate = model.get_layer('gated_fusion').gate.numpy()[0]
       print(f"Epoch {epoch}: Gate = {gate:.3f}")

   期望值: 0.80-0.90 之间

3. 对比新的Test MAE:
   目标: Test MAE < 18.79 (图基线)

预期结果:
──────────────────────────────────────────────────────────────────
如果加权组合(α=0.86)能达到 < 18.79:
  → 这个修复应该能达到类似或更好的结果
  → 预期 Test MAE: 17.5-18.5 范围

如果修复后效果还不够好:
──────────────────────────────────────────────────────────────────
尝试组合修复:

修复A: 禁用中期融合
  在训练配置中:
  model = make_model_multimodal_v5(
      use_middle_fusion=False,  # ← 添加这个
      ...
  )

修复B: 增加文本dropout
  在 _make_dense_multimodal_v5.py 中:
  text_emb = Dropout(0.5)(text_emb)  # 50% dropout

修复C: 使用ResidualFusion
  替换GatedFusion为ResidualFusion (见文档)

如何恢复原版本:
──────────────────────────────────────────────────────────────────
python apply_fusion_fix.py --restore

或手动恢复:
cp [备份文件] kgcnn/literature/DenseGNN/_multimodal_fusion.py

查看备份文件:
ls -lt kgcnn/literature/DenseGNN/_multimodal_fusion_backup_*.py

╔═══════════════════════════════════════════════════════════════════╗
║ 祝训练顺利! 如有问题请查看 training/fixes/fix_fusion_weights.md  ║
╚═══════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description='应用融合权重修复')
    parser.add_argument('--restore', action='store_true',
                       help='恢复到原版本(使用最新备份)')
    args = parser.parse_args()

    print("="*70)
    print("DenseGNN 融合权重修复工具")
    print("="*70)
    print()

    if args.restore:
        print("模式: 恢复原版本")
        print()
        success = restore_original()
    else:
        print("模式: 应用修复 (α=0.86)")
        print()
        success = apply_fix()

        if success:
            print()
            print_instructions()

    print()
    if success:
        print("✅ 操作成功完成!")
    else:
        print("❌ 操作失败")

    print("="*70)


if __name__ == '__main__':
    main()
