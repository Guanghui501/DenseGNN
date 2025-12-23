"""
深度诊断: 检查图-文本互补性

基于你的结果:
- Graph-only: 18.79 MAE
- Text-only: 25.58 MAE
- Multimodal: 20.06 MAE (比Graph差!)

这个脚本会回答:
1. 文本和图学到的是相同信息还是不同信息?
2. 简单线性组合能否打败18.79?
3. 问题是融合机制还是数据本身?
"""

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def diagnose_graph_text_complementarity(graph_emb, text_emb, labels,
                                        graph_baseline_mae=18.79,
                                        text_baseline_mae=25.58,
                                        multimodal_mae=20.06):
    """
    诊断图-文本互补性问题

    检查:
    1. 预测重叠度 (两个模态是否学到相同东西)
    2. 简单线性组合性能 (理论上限)
    3. 负向迁移检测
    """

    print("\n" + "="*80)
    print("图-文本互补性深度诊断")
    print("="*80)

    # 确保是numpy
    if tf.is_tensor(graph_emb):
        graph_emb = graph_emb.numpy()
    if tf.is_tensor(text_emb):
        text_emb = text_emb.numpy()
    if tf.is_tensor(labels):
        labels = labels.numpy()

    if len(labels.shape) > 1:
        labels = labels.flatten()

    # 分割数据
    n = len(labels)
    train_idx = int(0.8 * n)

    g_train, g_val = graph_emb[:train_idx], graph_emb[train_idx:]
    t_train, t_val = text_emb[:train_idx], text_emb[train_idx:]
    y_train, y_val = labels[:train_idx], labels[train_idx:]

    results = {}

    # ========================================================================
    # 测试1: 预测重叠度检测
    # ========================================================================
    print("\n[测试1] 预测重叠度检测")
    print("-" * 80)

    # 训练单独的图模型
    model_g = Ridge(alpha=1.0)
    model_g.fit(g_train, y_train)
    pred_g = model_g.predict(graph_emb)

    # 训练单独的文本模型
    model_t = Ridge(alpha=1.0)
    model_t.fit(t_train, y_train)
    pred_t = model_t.predict(text_emb)

    # 计算预测的相关性
    corr, p_value = pearsonr(pred_g, pred_t)

    print(f"图预测 vs 文本预测 相关性: {corr:.3f}")

    results['prediction_correlation'] = float(corr)

    if corr > 0.8:
        print("\n🔴 问题检测: 高度重叠 (correlation > 0.8)")
        print("   → 文本和图学到的是**相同的信息**")
        print("   → 文本没有提供新的补充信息")
        print("   → 建议: **使用纯图模型** (18.79 MAE)")
        results['diagnosis'] = 'high_overlap'

    elif corr > 0.5:
        print("\n⚠️  中度重叠 (0.5 < correlation < 0.8)")
        print("   → 文本和图有部分重叠，但也有独特信息")
        print("   → 多模态可能有小幅提升")
        results['diagnosis'] = 'medium_overlap'

    else:
        print("\n✅ 低重叠 (correlation < 0.5)")
        print("   → 文本和图学到了不同的信息")
        print("   → 理论上多模态应该能显著提升")
        results['diagnosis'] = 'low_overlap'

    # ========================================================================
    # 测试2: 简单线性组合性能 (理论上限)
    # ========================================================================
    print("\n[测试2] 简单线性组合性能测试")
    print("-" * 80)

    # 方法A: Concatenate + Ridge
    combined_train = np.concatenate([g_train, t_train], axis=-1)
    combined_val = np.concatenate([g_val, t_val], axis=-1)

    model_concat = Ridge(alpha=1.0)
    model_concat.fit(combined_train, y_train)
    pred_concat = model_concat.predict(combined_val)
    mae_concat = mean_absolute_error(y_val, pred_concat)

    # 方法B: 加权平均 (学习权重)
    # 学习最优alpha: final_pred = alpha * pred_g + (1-alpha) * pred_t
    best_alpha = None
    best_mae = float('inf')

    pred_g_val = model_g.predict(g_val)
    pred_t_val = model_t.predict(t_val)

    for alpha in np.linspace(0, 1, 101):
        pred_weighted = alpha * pred_g_val + (1 - alpha) * pred_t_val
        mae = mean_absolute_error(y_val, pred_weighted)
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    mae_weighted = best_mae

    # 方法C: Stacking (先预测，再组合预测)
    stacking_train = np.stack([model_g.predict(g_train), model_t.predict(t_train)], axis=1)
    stacking_val = np.stack([pred_g_val, pred_t_val], axis=1)

    model_stack = Ridge(alpha=1.0)
    model_stack.fit(stacking_train, y_train)
    pred_stack = model_stack.predict(stacking_val)
    mae_stack = mean_absolute_error(y_val, pred_stack)

    print(f"\n简单组合方法测试:")
    print(f"  方法A - Concat + Ridge:     {mae_concat:.2f} MAE")
    print(f"  方法B - 加权平均 (α={best_alpha:.2f}): {mae_weighted:.2f} MAE")
    print(f"  方法C - Stacking:            {mae_stack:.2f} MAE")

    print(f"\n与基线对比:")
    print(f"  Graph-only 基线:             {graph_baseline_mae:.2f} MAE")
    print(f"  你的Multimodal:              {multimodal_mae:.2f} MAE")

    best_simple_mae = min(mae_concat, mae_weighted, mae_stack)
    results['best_simple_combination'] = float(best_simple_mae)
    results['concat_mae'] = float(mae_concat)
    results['weighted_mae'] = float(mae_weighted)
    results['stacking_mae'] = float(mae_stack)
    results['optimal_alpha'] = float(best_alpha)

    # 分析
    if best_simple_mae < graph_baseline_mae:
        improvement = graph_baseline_mae - best_simple_mae
        print(f"\n✅ 简单组合能打败基线! (提升 {improvement:.2f} MAE)")
        print(f"   → 理论上限: {best_simple_mae:.2f} MAE")
        print(f"   → 你的模型: {multimodal_mae:.2f} MAE")
        print(f"   → 差距: {multimodal_mae - best_simple_mae:.2f} MAE")
        print("\n   🔴 问题诊断: **融合机制不当**")
        print("      简单的线性组合都比你的神经网络融合好!")
        print("      说明问题不在数据，而在融合方式。")
        results['can_improve'] = True
        results['problem'] = 'fusion_mechanism'

    else:
        gap = best_simple_mae - graph_baseline_mae
        print(f"\n❌ 简单组合打不败基线 (差 {gap:.2f} MAE)")
        print(f"   → 即使最优组合也只能达到: {best_simple_mae:.2f} MAE")
        print(f"   → Graph基线: {graph_baseline_mae:.2f} MAE")
        print("\n   🔴 问题诊断: **负向迁移**")
        print("      文本信息与图信息冲突，加入文本反而变差。")
        print("      即使用简单线性组合也无法提升。")
        results['can_improve'] = False
        results['problem'] = 'negative_transfer'

    # ========================================================================
    # 测试3: 特征重要性分析
    # ========================================================================
    print("\n[测试3] 特征重要性分析")
    print("-" * 80)

    # 使用Lasso看哪些特征重要
    model_lasso = Lasso(alpha=0.1)
    model_lasso.fit(combined_train, y_train)

    graph_dim = g_train.shape[1]
    text_dim = t_train.shape[1]

    graph_coef = model_lasso.coef_[:graph_dim]
    text_coef = model_lasso.coef_[graph_dim:]

    graph_importance = np.sum(np.abs(graph_coef))
    text_importance = np.sum(np.abs(text_coef))
    total_importance = graph_importance + text_importance

    graph_ratio = graph_importance / (total_importance + 1e-8)
    text_ratio = text_importance / (total_importance + 1e-8)

    print(f"线性模型中的特征重要性:")
    print(f"  图特征重要性: {graph_ratio:.1%}")
    print(f"  文本特征重要性: {text_ratio:.1%}")

    results['graph_importance_ratio'] = float(graph_ratio)
    results['text_importance_ratio'] = float(text_ratio)

    if text_ratio < 0.1:
        print("\n⚠️  文本特征几乎不重要 (<10%)")
        print("   → 线性模型主要依赖图特征")
        print("   → 文本对预测的贡献很小")

    # ========================================================================
    # 测试4: 残差分析 (文本能否预测图的错误?)
    # ========================================================================
    print("\n[测试4] 残差分析 - 文本能否修正图的错误?")
    print("-" * 80)

    # 图模型的残差
    graph_residuals = y_train - model_g.predict(g_train)

    # 用文本预测这些残差
    model_residual = Ridge(alpha=1.0)
    model_residual.fit(t_train, graph_residuals)

    # 在验证集上测试
    pred_g_val = model_g.predict(g_val)
    residual_correction = model_residual.predict(t_val)
    pred_corrected = pred_g_val + residual_correction

    mae_corrected = mean_absolute_error(y_val, pred_corrected)
    mae_graph_val = mean_absolute_error(y_val, pred_g_val)

    print(f"图模型 MAE: {mae_graph_val:.2f}")
    print(f"图+文本残差修正 MAE: {mae_corrected:.2f}")

    results['residual_correction_mae'] = float(mae_corrected)

    if mae_corrected < mae_graph_val:
        improvement = mae_graph_val - mae_corrected
        print(f"\n✅ 文本能修正图的错误! (提升 {improvement:.2f} MAE)")
        print("   → 文本包含图没有的补充信息")
        print("   → 应该使用残差融合方式")
        results['text_can_correct'] = True
    else:
        print(f"\n❌ 文本无法修正图的错误")
        print("   → 文本不包含补充信息")
        print("   → 或者文本的信息已经在图中了")
        results['text_can_correct'] = False

    # ========================================================================
    # 最终建议
    # ========================================================================
    print("\n" + "="*80)
    print("诊断结论与建议")
    print("="*80)

    if results['problem'] == 'negative_transfer':
        print("\n🔴 诊断: **负向迁移**")
        print("\n现象:")
        print(f"  - 即使最优线性组合 ({best_simple_mae:.2f}) 也比图基线差")
        print(f"  - 说明文本和图的信息互相冲突")
        print("\n**强烈建议: 使用纯图模型 (18.79 MAE)**")
        print("\n原因:")
        print("  - 文本虽然单独能预测 (25.58 MAE)")
        print("  - 但与图组合时产生负面效果")
        print("  - 可能的原因:")
        print("    * 文本描述的是宏观属性，图描述的是微观结构")
        print("    * 两个模态的信息不一致")
        print("    * 文本有系统性偏差或错误")

    elif results['problem'] == 'fusion_mechanism':
        print("\n🔴 诊断: **融合机制问题**")
        print("\n现象:")
        print(f"  - 简单线性组合能达到: {best_simple_mae:.2f} MAE < 18.79")
        print(f"  - 你的神经网络融合: {multimodal_mae:.2f} MAE > 18.79")
        print(f"  - 性能差距: {multimodal_mae - best_simple_mae:.2f} MAE")
        print("\n**建议: 修复融合机制**")
        print("\n具体行动:")

        print(f"\n1. 尝试简单线性融合 (最优α={best_alpha:.2f})")
        print("   实现:")
        print(f"   fused = {best_alpha:.2f} * graph_emb + {1-best_alpha:.2f} * text_emb")

        if results.get('text_can_correct', False):
            print("\n2. 使用残差融合 (推荐!)")
            print("   文本能修正图的错误，应该用这种方式:")
            print("   实现:")
            print("   graph_pred = graph_head(graph_emb)")
            print("   text_correction = text_head(text_emb)")
            print("   final_pred = graph_pred + 0.3 * text_correction")

        print("\n3. 检查当前的gate权重")
        print("   打印你模型的gate值，看是否合理")
        print(f"   理论最优值应该接近: {best_alpha:.2f}")

        print("\n4. 禁用中期融合")
        print("   use_middle_fusion=False")
        print("   只用late fusion可能更稳定")

    # 保存结果
    import json
    with open('./complementarity_diagnosis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n完整结果已保存到: ./complementarity_diagnosis.json")
    print("="*80 + "\n")

    return results


def plot_predictions_comparison(graph_pred, text_pred, labels, save_path='./predictions_comparison.png'):
    """可视化图和文本的预测对比"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Graph predictions vs labels
    ax = axes[0, 0]
    ax.scatter(labels, graph_pred, alpha=0.5, s=20)
    ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'r--', lw=2)
    ax.set_xlabel('True Labels')
    ax.set_ylabel('Graph Predictions')
    ax.set_title('Graph Predictions vs True Labels')
    ax.grid(True, alpha=0.3)

    # 2. Text predictions vs labels
    ax = axes[0, 1]
    ax.scatter(labels, text_pred, alpha=0.5, s=20)
    ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'r--', lw=2)
    ax.set_xlabel('True Labels')
    ax.set_ylabel('Text Predictions')
    ax.set_title('Text Predictions vs True Labels')
    ax.grid(True, alpha=0.3)

    # 3. Graph vs Text predictions
    ax = axes[1, 0]
    ax.scatter(graph_pred, text_pred, alpha=0.5, s=20)
    ax.set_xlabel('Graph Predictions')
    ax.set_ylabel('Text Predictions')
    ax.set_title('Graph vs Text Predictions\n(High correlation = high overlap)')
    ax.grid(True, alpha=0.3)

    # Add correlation
    from scipy.stats import pearsonr
    corr, _ = pearsonr(graph_pred, text_pred)
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Residuals
    ax = axes[1, 1]
    graph_residuals = labels - graph_pred
    text_residuals = labels - text_pred
    ax.scatter(graph_residuals, text_residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--', lw=1)
    ax.axvline(x=0, color='r', linestyle='--', lw=1)
    ax.set_xlabel('Graph Residuals')
    ax.set_ylabel('Text Residuals')
    ax.set_title('Residuals Comparison\n(Quadrants show where each modality fails)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"预测对比图已保存到: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("""
这个脚本需要你提供:
- graph_emb: 图嵌入
- text_emb: 文本嵌入
- labels: 真实标签

使用方法:

from diagnostics.deep_complementarity_check import diagnose_graph_text_complementarity

# 从你的模型提取嵌入
graph_emb = model.get_layer('graph_projection').output  # 或者其他方式提取
text_emb = model.get_layer('text_projection').output

# 运行诊断
results = diagnose_graph_text_complementarity(
    graph_emb=graph_emb,
    text_emb=text_emb,
    labels=labels,
    graph_baseline_mae=18.79,
    text_baseline_mae=25.58,
    multimodal_mae=20.06
)

# 查看结果
print(results)
""")
