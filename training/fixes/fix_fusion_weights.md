# 修复融合权重到最优值 α=0.86

## 问题诊断

**发现:**
- 最优组合权重: α = 0.86 (图86%, 文本14%)
- 当前模型很可能使用了不正确的融合权重
- 导致 Multimodal (20.06) > Graph-only (18.79)

## 当前GatedFusion的问题

当前实现（`_multimodal_fusion.py` 第292-304行）:

```python
# 计算两个独立的gate
gate_g = self.gate_graph_dense2(gate_g)  # sigmoid输出
gate_t = self.gate_text_dense2(gate_t)   # sigmoid输出

# 归一化
gate_sum = gate_g + gate_t + 1e-8
gate_g = gate_g / gate_sum
gate_t = gate_t / gate_sum
```

**问题:**
- 两个gate都是sigmoid(0-1)，初始时可能都接近0.5
- 归一化后，如果 gate_g ≈ gate_t，则融合比例 ≈ 50:50
- **没有偏向图的初始化，与最优的86:14相差甚远!**

## 修复方案 (按推荐顺序)

### 🌟 修复#1: 简化为单一gate，初始化为0.86 (最推荐!)

**实现最简单，效果最好，直接使用学到的最优权重**

编辑 `kgcnn/literature/DenseGNN/_multimodal_fusion.py`，替换整个`GatedFusion`类:

```python
class GatedFusion(layers.Layer):
    """Gated fusion with single learnable weight.

    Uses optimal weight discovered from diagnostics: α=0.86 for graph, 0.14 for text.
    fused = gate * graph_transformed + (1-gate) * text_transformed
    where gate is initialized to 0.86 and can be fine-tuned during training.
    """

    def __init__(self, graph_dim=64, text_dim=64, output_dim=64, dropout=0.1,
                 initial_gate=0.86, **kwargs):  # ← 新增initial_gate参数
        super().__init__(**kwargs)
        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.initial_gate = initial_gate  # ← 保存初始gate值

    def build(self, input_shape):
        # 单一可学习的gate权重，初始化为0.86
        self.gate = self.add_weight(
            name='fusion_gate',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.initial_gate),  # ← 0.86
            trainable=True,  # 可以在训练中微调
            constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0)  # 限制在[0,1]
        )

        # Feature transformation
        self.graph_transform = layers.Dense(self.output_dim)
        self.text_transform = layers.Dense(self.output_dim)

        # Fusion transformation
        self.fusion_dense = layers.Dense(self.output_dim)
        self.fusion_norm = layers.LayerNormalization()
        self.fusion_activation = layers.Activation('relu')
        self.fusion_dropout = layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs, training=None):
        graph_feat, text_feat = inputs

        # Transform features
        graph_transformed = self.graph_transform(graph_feat)
        text_transformed = self.text_transform(text_feat)

        # Simple weighted fusion: α * graph + (1-α) * text
        fused = self.gate * graph_transformed + (1 - self.gate) * text_transformed

        # Final transformation
        fused = self.fusion_dense(fused)
        fused = self.fusion_norm(fused)
        fused = self.fusion_activation(fused)
        fused = self.fusion_dropout(fused, training=training)

        return fused

    def get_config(self):
        config = super().get_config()
        config.update({
            'graph_dim': self.graph_dim,
            'text_dim': self.text_dim,
            'output_dim': self.output_dim,
            'dropout': self.dropout_rate,
            'initial_gate': self.initial_gate
        })
        return config
```

**如何应用:**
1. 替换 `_multimodal_fusion.py` 中的 `GatedFusion` 类
2. 重新训练模型
3. 检查训练日志中的gate值

**预期结果:**
- Gate从0.86开始，训练中可以微调
- 应该能接近或打败18.79 MAE
- 如果加权组合能达到 < 18.79，这个方法应该能达到类似效果

---

### 修复#2: 保留双gate，但调整bias初始化

**如果你想保持当前架构，只调整初始化**

编辑 `_multimodal_fusion.py` 第258-279行:

```python
def build(self, input_shape):
    # Gate for graph - 初始化为更大的bias，让它倾向输出更大值
    self.gate_graph_dense1 = layers.Dense(self.graph_dim // 2, activation='relu')
    self.gate_graph_dropout = layers.Dropout(self.dropout_rate)

    # 关键修改: 添加positive bias让graph gate初始更大
    self.gate_graph_dense2 = layers.Dense(
        1,
        activation='sigmoid',
        bias_initializer=tf.keras.initializers.Constant(2.0)  # ← 添加这个
    )

    # Gate for text - 初始化为更小的bias，让它倾向输出更小值
    self.gate_text_dense1 = layers.Dense(self.text_dim // 2, activation='relu')
    self.gate_text_dropout = layers.Dropout(self.dropout_rate)

    # 关键修改: 添加negative bias让text gate初始更小
    self.gate_text_dense2 = layers.Dense(
        1,
        activation='sigmoid',
        bias_initializer=tf.keras.initializers.Constant(-2.0)  # ← 添加这个
    )

    # ... 其余代码不变 ...
```

**原理:**
- sigmoid(2.0) ≈ 0.88
- sigmoid(-2.0) ≈ 0.12
- 归一化后: 0.88 / (0.88 + 0.12) = 0.88, 接近最优的0.86!

**预期效果:**
- 初始融合比例 ≈ 88:12 (接近最优的86:14)
- 训练中可以调整

---

### 修复#3: 禁用中期融合 (配合修复#1或#2)

**中期融合可能给文本太多影响**

编辑你的训练配置或超参数文件:

```python
model = make_model_multimodal_v5(
    use_middle_fusion=False,  # ← 改为False
    late_fusion_type='gated',
    ...
)
```

**原因:**
- 中期融合在每一层都注入文本特征
- Dense连接会累积这些文本特征
- 可能导致文本影响过大

**建议:**
- 先应用修复#1或#2
- 如果还不够好，再禁用中期融合

---

### 修复#4: 不对称dropout (给文本更高dropout)

**进一步降低文本的影响**

编辑 `_make_dense_multimodal_v5.py`:

```python
# 在text projection之后 (第129行附近)
text_projection = ProjectionHead(
    embedding_dim=text_embedding_dim,
    projection_dim=text_projection_dim,
    dropout=0.5  # ← 从0.1提高到0.5
)
text_emb = text_projection(text_input)

# 可选: 添加额外的dropout
text_emb = tf.keras.layers.Dropout(0.3)(text_emb)  # ← 新增

# Graph保持较低dropout
graph_projection = ProjectionHead(
    embedding_dim=graph_out.shape[-1],
    projection_dim=graph_projection_dim,
    dropout=0.1  # ← 保持低dropout
)
```

**预期效果:**
- 训练时文本特征被随机丢弃50-70%
- 强制模型更依赖图特征
- 防止过拟合到文本

---

### 修复#5: 残差融合 (如果诊断显示文本能修正错误)

**将文本作为图的小幅修正，而不是平等融合**

创建新的融合层 `ResidualFusion`:

```python
class ResidualFusion(layers.Layer):
    """Residual fusion: graph主预测 + text小幅修正"""

    def __init__(self, graph_dim=64, text_dim=64, correction_weight=0.14, **kwargs):
        super().__init__(**kwargs)
        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.correction_weight = correction_weight  # 0.14 = 1 - 0.86

    def build(self, input_shape):
        # 图的主预测头
        self.graph_head = layers.Dense(1, name='graph_prediction')

        # 文本的修正头
        self.text_correction_head = layers.Dense(1, name='text_correction')

        super().build(input_shape)

    def call(self, inputs, training=None):
        graph_emb, text_emb = inputs

        # 图的主预测
        graph_pred = self.graph_head(graph_emb)

        # 文本的修正
        text_correction = self.text_correction_head(text_emb)

        # 残差组合
        final_pred = graph_pred + self.correction_weight * text_correction

        return final_pred
```

**使用方法:**

在 `_make_dense_multimodal_v5.py` 中，替换late fusion部分:

```python
# 替换原来的GatedFusion
from ._multimodal_fusion import ResidualFusion

residual_fusion = ResidualFusion(
    graph_dim=graph_projection_dim,
    text_dim=text_projection_dim,
    correction_weight=0.14  # 1 - 0.86
)
out = residual_fusion([graph_emb, text_emb])
```

**预期效果:**
- 图占主导 (86%)
- 文本只提供小幅修正 (14%)
- 架构明确反映最优权重

---

## 推荐的实施顺序

### 第1步: 快速修复 (今天, 30分钟)

**应用修复#1 (简化为单一gate)**
- 替换 `GatedFusion` 类
- 重新训练
- **预期: MAE应该显著改善，接近18.79或更好**

### 第2步: 如果修复#1效果不够好 (明天, 1小时)

**组合修复:**
- 修复#1 (单一gate初始化0.86)
- +修复#3 (禁用中期融合)
- +修复#4 (不对称dropout)

### 第3步: 如果还不行，尝试架构调整

**应用修复#5 (残差融合)**
- 更彻底的架构改变
- 明确图为主、文本为辅

---

## 如何验证修复是否有效

### 训练时监控:

1. **打印gate值** (如果用修复#1):

```python
# 在训练循环中
if epoch % 10 == 0:
    gate_value = model.get_layer('gated_fusion').gate.numpy()
    print(f"Epoch {epoch}: Fusion gate = {gate_value[0]:.3f}")
```

**期望值:** 应该在0.80-0.90之间

2. **对比MAE**:

```python
# 每个epoch记录
train_mae = ...
val_mae = ...

# 目标: val_mae < 18.79
```

### 测试结果判断:

| Test MAE | 诊断 | 下一步 |
|----------|------|--------|
| **< 18.79** | ✅ 成功! | 多模态比图基线好 |
| **≈ 18.79** | ⚠️ 勉强 | 可接受，但提升有限 |
| **18.79-19.5** | ⚠️ 小幅改善 | 尝试组合修复 |
| **> 19.5** | ❌ 还不够 | 尝试修复#5或其他架构 |

---

## 关键问题

**你还没告诉我: 加权组合(α=0.86)的MAE是多少?**

这个数字非常关键:
- 如果 < 18.79: 说明修复#1应该能work
- 如果 > 18.79: 说明即使最优权重也无法打败基线，可能需要其他方法

**请告诉我这个值，我可以更精确地指导你!**

---

## 总结

**你的发现:**
- 最优权重 α = 0.86 (图86%, 文本14%)
- 说明文本应该是次要模态

**最可能的问题:**
- 当前GatedFusion初始化不当，没有偏向图
- 可能给了文本太多权重 (>14%)

**推荐修复:**
1. **立即:** 应用修复#1 (单一gate，初始0.86)
2. **如果不够:** 加上修复#3 (禁用中期融合)
3. **如果还不够:** 加上修复#4 (不对称dropout)

**预期结果:**
- 修复后应该接近或打败 18.79 MAE
- 如果加权组合能到 < 18.79，修复#1应该能达到类似效果

立即应用修复#1，重新训练，看看效果! 🚀
