#!/bin/bash
# 应用融合权重修复

echo "=================================="
echo "应用融合权重修复 (α=0.86)"
echo "=================================="

# 备份原文件
echo ""
echo "[1/3] 备份原始文件..."
cp kgcnn/literature/DenseGNN/_multimodal_fusion.py \
   kgcnn/literature/DenseGNN/_multimodal_fusion_backup_$(date +%Y%m%d_%H%M%S).py

echo "✓ 备份完成"

# 替换为修复版本
echo ""
echo "[2/3] 应用修复..."
cp kgcnn/literature/DenseGNN/_multimodal_fusion_fixed.py \
   kgcnn/literature/DenseGNN/_multimodal_fusion.py

echo "✓ 修复已应用"

# 提示
echo ""
echo "[3/3] 下一步操作:"
echo "=================================="
echo ""
echo "修复已应用! 修改内容:"
echo "  - GatedFusion gate初始化为 0.86 (图86%, 文本14%)"
echo "  - 简化为单一gate权重"
echo "  - 新增 ResidualFusion 类供备选"
echo ""
echo "现在请:"
echo "  1. 重新训练模型"
echo "  2. 监控gate值 (应该在0.80-0.90之间)"
echo "  3. 对比新的Test MAE与18.79基线"
echo ""
echo "如果需要恢复原版本:"
echo "  找到最新的备份文件:"
echo "  ls -lt kgcnn/literature/DenseGNN/_multimodal_fusion_backup_*.py | head -1"
echo ""
echo "可选修复 (如果效果还不够好):"
echo "  - 禁用中期融合: use_middle_fusion=False"
echo "  - 增加文本dropout: 在text_projection后加Dropout(0.5)"
echo ""
echo "=================================="
