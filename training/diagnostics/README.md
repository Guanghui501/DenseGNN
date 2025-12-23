# Multimodal Fusion Diagnostics

## Overview

This diagnostic toolkit helps identify why adding text modality to DenseGNN **decreases** performance instead of improving it.

### The Problem

You've discovered that:
- ✅ **DenseGNN Lite (pure graph, 2-way)**: Good performance
- ❌ **DenseGNN v5 multimodal (graph+text, 2-way)**: Worse than pure graph!
- ❌ **DenseGNN v6 multimodal (graph+text, 3-way)**: Also worse performance

This indicates the problem is **not the architecture** but something fundamental about how graph and text modalities are being fused.

### Potential Root Causes

1. **Feature Scale Imbalance** 🔴 (Most likely)
   - Graph embeddings and BERT text embeddings have different magnitude ranges
   - Fusion will be dominated by whichever modality has larger values
   - Example: Graph std=5.0, Text std=1.0 → Graph dominates 5:1

2. **Text Noise/Quality Issues** 🔴
   - Auto-generated text descriptions may be inaccurate or inconsistent
   - Text may not align well with graph structure
   - High variance across samples suggests noisy text

3. **Over-reliance on Text** 🔴
   - Model may learn shortcuts from text patterns
   - Text could contain label leakage
   - Gate weights < 0.3 indicates text dominance

4. **Inappropriate Fusion Method**
   - Simple gated fusion may be insufficient
   - Dropout settings may not be optimal
   - Projection dimensions (128) might lose information

5. **Frozen BERT Encoder**
   - Pre-trained BERT not adapted to materials science domain
   - Generic embeddings may not capture relevant features

6. **Training Strategy Issues**
   - Learning rates not optimized for multimodal
   - Missing warmup or curriculum learning
   - Different modalities may need different learning rates

7. **Data Alignment Problems**
   - Semantic gap between microscopic graph structure and macroscopic text descriptions
   - Text describes bulk properties, graph shows atomic arrangement

---

## Quick Start

### Option 1: Quick Diagnostic Check (5 minutes)

Run a fast diagnostic on one batch to immediately see if you have feature imbalance:

```bash
python training/diagnostics/run_diagnostics_v5.py \
    --model_path ./models/v5_model.h5 \
    --data_path ./data/test_data.pkl \
    --quick
```

**Output:**
```
QUICK DIAGNOSTIC CHECK
============================================================
Graph embedding std:  4.823
Text embedding std:   0.976
Std ratio (G/T):      4.94x

🔴 WARNING: Feature scale imbalance detected!
```

**What this tells you:** If ratio > 3x, you have severe feature scale imbalance. Graph and text embeddings are on different scales, so fusion won't work properly.

---

### Option 2: Full Diagnostic Report (30 minutes)

Run comprehensive diagnostics on 100 batches to generate detailed analysis:

```bash
python training/diagnostics/run_diagnostics_v5.py \
    --model_path ./models/v5_model.h5 \
    --data_path ./data/test_data.pkl \
    --num_batches 100 \
    --output_dir ./diagnostic_results
```

**Outputs:**
- `diagnostic_results/diagnostic_report.json` - Full analysis
- `diagnostic_results/plots/embedding_statistics.png` - Embedding scale plots
- `diagnostic_results/plots/gate_weights.png` - Fusion gate evolution
- `diagnostic_results/tensorboard/` - TensorBoard logs

**View in TensorBoard:**
```bash
tensorboard --logdir ./diagnostic_results/tensorboard
# Open http://localhost:6006
```

---

### Option 3: Train with Diagnostics (Full Training)

Integrate diagnostics into your training loop for real-time monitoring.

#### Step 1: Modify Model to Expose Embeddings

Edit: `kgcnn/literature/DenseGNN/_make_dense_multimodal_v5.py`

**Line 198, change from:**
```python
return ks.Model(inputs=input_list, outputs=out, name=name)
```

**To:**
```python
return ks.Model(
    inputs=input_list,
    outputs={
        'prediction': out,
        'graph_emb': graph_emb,
        'text_emb': text_emb
    },
    name=name
)
```

#### Step 2: Add Diagnostics to Training Script

At the top of your training script:

```python
from training.diagnostics.multimodal_diagnostics import MultimodalDiagnostics

# Initialize diagnostics
diagnostics = MultimodalDiagnostics(
    log_dir='./logs/v5_diagnostics',
    enable_tb=True
)
```

#### Step 3: Update Training Loop

```python
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            # Get outputs (now a dict)
            outputs = model(batch, training=True)
            predictions = outputs['prediction']
            graph_emb = outputs['graph_emb']
            text_emb = outputs['text_emb']

            loss = loss_fn(labels, predictions)

        # Gradient update
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Log diagnostics every 20 batches
        if batch_idx % 20 == 0:
            diagnostics.log_embeddings(graph_emb, text_emb, epoch, batch_idx)
            diagnostics.log_loss(loss, epoch, with_text=True)

    # End of epoch: log gate weights
    gate_layer = model.get_layer('gated_fusion')
    diagnostics.log_fusion_gates(gate_layer.gate.numpy(), epoch)

# After training: generate report
diagnostics.generate_report(save_plots=True)
```

---

## Understanding Diagnostic Output

### 1. Feature Scale Imbalance Check

```
[1] FEATURE SCALE IMBALANCE CHECK
────────────────────────────────────────────────────────────────────────────────
Graph embedding std:  5.234
Text embedding std:   1.127
Std ratio (G/T):      4.64x
Norm ratio (G/T):     3.82x

🔴 PROBLEM DETECTED: Severe feature scale imbalance!
   → Graph and text embeddings have incompatible scales
   → Fusion will be dominated by the larger-scale modality

   Solutions:
   - Add LayerNormalization before fusion
   - Use separate projection heads with normalization
   - Scale text embeddings: text_emb = text_emb * scale_factor
```

**What this means:**
- Graph embeddings have standard deviation 5.2, text has 1.1
- Graph values are ~4.6x larger than text
- During fusion, graph features will dominate, text will be ignored
- Model is effectively learning from graph only, text adds noise

**How to fix:**
```python
# Option 1: Add layer normalization
from tensorflow.keras.layers import LayerNormalization

graph_emb = LayerNormalization()(graph_emb)
text_emb = LayerNormalization()(text_emb)

# Option 2: Manual scaling
text_emb = text_emb * scale_factor  # e.g., 4.64

# Option 3: L2 normalization
graph_emb = tf.nn.l2_normalize(graph_emb, axis=-1)
text_emb = tf.nn.l2_normalize(text_emb, axis=-1)
```

---

### 2. Text Embedding Quality Check

```
[2] TEXT EMBEDDING QUALITY CHECK
────────────────────────────────────────────────────────────────────────────────
Text embedding noise score: 0.673

🔴 PROBLEM DETECTED: High text embedding variance!
   → Text embeddings are inconsistent across samples
   → May indicate poor text quality or encoding issues

   Solutions:
   - Review text descriptions for quality
   - Fine-tune BERT encoder (if frozen)
   - Add text embedding dropout for regularization
```

**What this means:**
- Text embeddings have high variance across different batches
- Suggests text quality is inconsistent or noisy
- Auto-generated descriptions may be unreliable

**How to fix:**
```python
# Option 1: Fine-tune BERT (if currently frozen)
text_encoder.trainable = True

# Option 2: Add dropout to text branch
text_emb = tf.keras.layers.Dropout(0.3)(text_emb, training=training)

# Option 3: Review and clean text data
# Check text_descriptions.csv for quality issues
```

---

### 3. Text Over-Reliance Check

```
[3] TEXT OVER-RELIANCE CHECK
────────────────────────────────────────────────────────────────────────────────
Average gate weight (0=text, 1=graph): 0.234
Gate weight trend: -0.0123

🔴 PROBLEM DETECTED: Model over-relies on text!
   → Gate weights heavily favor text modality
   → Model may ignore graph structure

   Solutions:
   - Increase graph embedding dimension
   - Add modality-specific dropout (higher for text)
   - Use contrastive learning to align modalities
```

**What this means:**
- Gate weight of 0.234 means model uses 23% graph, 77% text
- Model is overfitting to text patterns
- Graph information being ignored despite being more reliable

**How to fix:**
```python
# Option 1: Increase graph projection dimension
graph_projection_dim = 256  # was 128
text_projection_dim = 128   # keep lower

# Option 2: Add text dropout
text_emb = tf.keras.layers.Dropout(0.3)(text_emb)  # 30% dropout on text
graph_emb = tf.keras.layers.Dropout(0.1)(graph_emb)  # 10% on graph

# Option 3: Initialize gate to favor graph
# In GatedFusion, change initial gate value
self.gate = self.add_weight(
    initializer=tf.keras.initializers.Constant(0.7),  # Favor graph initially
    ...
)
```

---

### 4. Visualizations

#### Embedding Statistics Plot

![Embedding Statistics](plots/embedding_statistics.png)

- **Top left:** Standard deviation over time
  - Both lines should be similar height
  - Large gap = scale imbalance

- **Top right:** L2 norm over time
  - Should be similar magnitude
  - Diverging lines = one modality dominating

- **Bottom left:** Std ratio over time
  - Should hover around 1.0 (red line)
  - Above 3.0 or below 0.33 = severe imbalance

- **Bottom right:** Norm ratio
  - Should be close to 1.0
  - Indicates relative magnitudes

#### Gate Weights Plot

![Gate Weights](plots/gate_weights.png)

- **Green zone (0.3-0.7):** Healthy balanced fusion
- **Below 0.3:** Over-reliance on text (🔴 problem)
- **Above 0.7:** Text being ignored (⚠️ investigate text quality)
- **Trend:** Should be relatively stable
  - Strong downward trend = model learning to ignore graph
  - Strong upward trend = model learning text is noisy

---

## Recommended Fixes

### Fix #1: Add Layer Normalization (Try this first!)

**Rationale:** Most likely cause is feature scale imbalance

**Implementation:**
Edit `kgcnn/literature/DenseGNN/_make_dense_multimodal_v5.py`, around line 177:

```python
# After projections, add normalization
from tensorflow.keras.layers import LayerNormalization

graph_projection = ProjectionHead(...)
graph_emb = graph_projection(graph_out)
graph_emb = LayerNormalization(name='graph_norm')(graph_emb)  # ← Add this

text_projection = ProjectionHead(...)
text_emb = text_projection(text_input)
text_emb = LayerNormalization(name='text_norm')(text_emb)  # ← Add this

# Now fusion will receive balanced inputs
```

**Expected result:** Gate weights should move closer to 0.5, performance should improve

---

### Fix #2: Add Modality-Specific Dropout

**Rationale:** Prevent overfitting to text shortcuts

**Implementation:**

```python
# In _make_dense_multimodal_v5.py, before fusion:

text_emb = tf.keras.layers.Dropout(0.3)(text_emb)  # Higher dropout for text
graph_emb = tf.keras.layers.Dropout(0.1)(graph_emb)  # Lower for graph
```

**Expected result:** Model forced to use graph features, can't rely solely on text

---

### Fix #3: Ablation Study - Train Without Text

**Rationale:** Confirm text is actually helpful

**Implementation:**

```python
# Create two models:
model_no_text = make_model_multimodal_v5(..., use_middle_fusion=False, late_fusion_type='none')
model_with_text = make_model_multimodal_v5(..., use_middle_fusion=True, late_fusion_type='gated')

# Train both and compare
```

**Expected result:** With fixes, multimodal should outperform graph-only

---

### Fix #4: Fine-tune BERT Encoder

**Rationale:** Generic BERT may not capture materials science semantics

**Implementation:**

```python
# In your training script, unfreeze BERT
text_encoder.trainable = True

# Use different learning rate for text encoder
optimizer_graph = Adam(learning_rate=1e-4)
optimizer_text = Adam(learning_rate=1e-5)  # Lower LR for BERT fine-tuning
```

**Expected result:** Text embeddings better aligned with task

---

### Fix #5: L2 Normalization Before Fusion

**Rationale:** Force all embeddings to unit norm

**Implementation:**

```python
# Instead of LayerNorm, use L2 normalization
graph_emb = tf.nn.l2_normalize(graph_emb, axis=-1)
text_emb = tf.nn.l2_normalize(text_emb, axis=-1)

# Now both have norm=1.0, equal contribution to fusion
```

**Expected result:** Perfect scale balance, gate weights around 0.5

---

## Files Overview

```
training/diagnostics/
├── README.md                              ← You are here
├── multimodal_diagnostics.py              ← Core diagnostic toolkit
├── run_diagnostics_v5.py                  ← Ready-to-run diagnostic script
├── train_v5_with_diagnostics.py           ← Training integration examples
├── example_integration_v5.py              ← Code examples
└── /tmp/diagnose_multimodal.py            ← Analysis summary (read-only)
```

**Main files:**
- `multimodal_diagnostics.py` - The diagnostic toolkit (import this)
- `run_diagnostics_v5.py` - Run diagnostics on existing model
- `train_v5_with_diagnostics.py` - Integration guide

---

## FAQ

**Q: Do I need to retrain my model to use diagnostics?**

A: No! Use `run_diagnostics_v5.py` with your existing trained model.

**Q: The script says it can't extract embeddings from my model. What do I do?**

A: The model needs to expose embeddings. Either:
1. Use the wrapper in `run_diagnostics_v5.py` (automatic, may not work for all models)
2. Modify your model to output embeddings (recommended, see Step 1 above)

**Q: Can I use this for DenseGNN v6 (3-way)?**

A: Yes! The same principles apply. Just modify the model builder to expose embeddings.

**Q: What if diagnostics show NO problems?**

A: If scales are balanced, text quality is good, and gates are healthy (0.3-0.7), but performance still suffers, the issue may be:
- Semantic misalignment between graph and text
- Text doesn't contain useful information for the task
- Dataset-specific issues (label leakage, data quality)

**Q: Can I use this during training or only after?**

A: Both! For best results, integrate into training loop (Option 3) to monitor in real-time.

---

## Next Steps

1. **Run quick diagnostic** on your current v5 model
2. **Identify the problem** from diagnostic report
3. **Apply the corresponding fix** (try Fix #1 first - layer normalization)
4. **Re-train and re-evaluate**
5. **Compare results** - multimodal should now beat graph-only!

---

## Support

If you encounter issues:
1. Check that your model outputs embeddings (see Step 1)
2. Verify data format matches expected inputs
3. Review example integrations in `example_integration_v5.py`

Good luck diagnosing and fixing your multimodal fusion! 🔬🔧
