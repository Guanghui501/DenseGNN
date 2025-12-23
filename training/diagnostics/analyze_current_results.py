"""
Analysis of Current Experimental Results

Based on your results:
- DenseGNN Baseline (pure GNN): Test MAE = 18.79
- DenseGNN Multimodal V5 (with LayerNorm): Test MAE = 20.06
- Multimodal is WORSE by 1.27 MAE (6.8%)

This script provides deeper analysis and alternative solutions.
"""

import numpy as np
import tensorflow as tf
import os


def analyze_why_layernorm_didnt_help():
    """
    LayerNorm fixed feature scale imbalance, but performance is still worse.
    This means there are OTHER problems beyond scale imbalance.
    """

    print("\n" + "="*80)
    print("WHY LAYERNORM DIDN'T FIX THE PROBLEM")
    print("="*80)

    print("""
Observation:
  - Pure GNN: MAE = 18.79
  - Multimodal (with LayerNorm): MAE = 20.06
  - Difference: +1.27 MAE (6.8% worse)

Analysis:
  LayerNorm ensures graph_emb and text_emb have similar scales (mean=0, var=1).
  This fixes SCALE IMBALANCE but doesn't address:

  🔴 Problem 1: TEXT QUALITY
     → If text contains noise or irrelevant information, balancing scales just
       makes the model equally sensitive to both GOOD graph features and
       BAD text features.

  🔴 Problem 2: TEXT-TASK MISALIGNMENT
     → Text may describe properties unrelated to your prediction target.
       Example: Text says "this material is blue and shiny"
               Target: band gap energy
               → Color information is useless for band gap!

  🔴 Problem 3: OVERFITTING TO TEXT PATTERNS
     → Text may contain spurious correlations or patterns that don't
       generalize. Model learns text shortcuts instead of real physics.

  🔴 Problem 4: LABEL LEAKAGE
     → Text descriptions may accidentally mention or hint at target values.
       Example: "This material has a high formation energy"
               Target: formation energy
               → Model just reads "high" from text, ignores graph!

  🔴 Problem 5: POOR FUSION MECHANISM
     → Simple gated fusion (graph_emb * gate + text_emb * (1-gate)) may be
       insufficient. Need more sophisticated fusion (attention, cross-modal).

  🔴 Problem 6: TRAINING STRATEGY
     → Text and graph may need different learning rates, warmup schedules.
       Current strategy may optimize poorly for multimodal setting.
""")

    print("\nConclusion:")
    print("  LayerNorm is NECESSARY but NOT SUFFICIENT.")
    print("  We need to diagnose which of the 6 problems above is the root cause.")
    print("="*80 + "\n")


def provide_next_diagnostic_steps():
    """Provide concrete next steps to diagnose the problem."""

    print("\n" + "="*80)
    print("NEXT DIAGNOSTIC STEPS - PRIORITY ORDER")
    print("="*80)

    print("""
STEP 1: CHECK TEXT QUALITY (HIGHEST PRIORITY)
──────────────────────────────────────────────

Run text quality diagnostics to check if text is actually useful:

```bash
python training/diagnostics/check_text_data.py \\
    --data_path ./data/your_dataset.pkl \\
    --model_path ./models/v5_with_layernorm.h5
```

This will check:
  ✓ Does text-only model perform better than random?
  ✓ Are text embeddings diverse or all similar?
  ✓ Do similar texts have similar labels? (noise check)
  ✓ Does text contain label leakage?

Expected output:
  If text is GOOD:
    - Text-only MAE should be < 40 (reasonable, even if worse than graph)
    - Neighbor consistency should be better than random
    - No label leakage detected

  If text is BAD (likely in your case):
    - Text-only MAE > 50 (very poor)
    - Neighbor consistency similar to random (text is noise)
    - High text-label correlation (possible leakage)


STEP 2: INSPECT YOUR TEXT DATA (CRITICAL!)
───────────────────────────────────────────

Manually review 10-20 text descriptions and ask:

  1. Is the text RELEVANT to the prediction target?
     Example: If predicting formation energy, does text mention:
              ✓ Stability, bonding, structure
              ✗ Color, density, applications (unless relevant)

  2. Is the text ACCURATE?
     Example: "This is a stable compound" - is it actually stable?

  3. Does text VARY across samples?
     Example: All texts say "This is a crystalline material" → uninformative

  4. Does text LEAK the label?
     Example: "High formation energy material" when target is formation energy

  5. Is text GENERATED or HUMAN-WRITTEN?
     Generated text often has artifacts, repetitive patterns, or errors


STEP 3: RUN ABLATION EXPERIMENTS
─────────────────────────────────

Test these configurations to isolate the problem:

  Experiment A: Late Fusion Only (No Middle Fusion)
  ───────────────────────────────────────────────────
  ```python
  model = make_model_multimodal_v5(
      use_middle_fusion=False,  # Disable middle fusion
      late_fusion_type='gated',
      ...
  )
  ```

  Expected: If middle fusion is the problem, this should improve.


  Experiment B: Increase Graph Capacity
  ──────────────────────────────────────
  ```python
  model = make_model_multimodal_v5(
      graph_projection_dim=256,  # Increase from 128
      text_projection_dim=64,    # Decrease text to 64
      ...
  )
  ```

  Expected: If graph needs more capacity to compete with text, this helps.


  Experiment C: Heavy Text Dropout
  ─────────────────────────────────
  ```python
  # In _make_dense_multimodal_v5.py, after text projection:
  text_emb = Dropout(0.5)(text_emb, training=training)  # 50% dropout!
  ```

  Expected: If model overfits to text, heavy dropout forces it to use graph.


  Experiment D: Text-Only Baseline
  ─────────────────────────────────
  Train a simple MLP on BERT text embeddings only:

  ```python
  text_input = Input(shape=(768,))
  x = Dense(128, activation='relu')(text_input)
  x = Dropout(0.3)(x)
  output = Dense(1)(x)
  model = Model(text_input, output)
  ```

  Expected: If text-only MAE >> 30, text is not useful for this task.


  Experiment E: Freeze GNN, Train Only Fusion
  ────────────────────────────────────────────
  ```python
  # Freeze graph layers
  for layer in model.layers:
      if 'graph' in layer.name or 'gin' in layer.name:
          layer.trainable = False

  # Only train fusion and text encoder
  ```

  Expected: If this works, problem is in joint training optimization.


STEP 4: CHECK FOR LABEL LEAKAGE IN TEXT
────────────────────────────────────────

Critical check - does text accidentally reveal target values?

  Manual check:
    1. Print 10 text descriptions for lowest target values
    2. Print 10 text descriptions for highest target values
    3. Look for patterns:
       - Do high-value texts mention "high", "large", "strong"?
       - Do low-value texts mention "low", "small", "weak"?
       - Do texts describe the exact property you're predicting?

  Automated check:
    ```python
    from training.diagnostics.diagnose_text_quality import TextQualityDiagnostics

    diagnostics = TextQualityDiagnostics()

    # Extract text embeddings from your dataset
    text_embeddings = []
    labels = []
    for batch in dataset:
        text_emb = bert_encoder(batch['text'])
        text_embeddings.append(text_emb)
        labels.append(batch['label'])

    text_embeddings = np.concatenate(text_embeddings)
    labels = np.concatenate(labels)

    # Check correlation
    report = diagnostics.analyze_text_label_correlation(text_embeddings, labels)

    if report['max_correlation'] > 0.5:
        print("🔴 LABEL LEAKAGE DETECTED!")
    ```


STEP 5: TRY ALTERNATIVE FUSION METHODS
───────────────────────────────────────

If text quality is OK but fusion doesn't work, try:

  Option A: Attention-Based Fusion
  ─────────────────────────────────
  Replace GatedFusion with cross-attention:

  ```python
  from tensorflow.keras.layers import MultiHeadAttention

  # Graph attends to text
  attn_layer = MultiHeadAttention(num_heads=4, key_dim=32)
  graph_emb_enhanced = attn_layer(
      query=graph_emb,
      key=text_emb,
      value=text_emb
  )

  # Combine
  fused = Concatenate()([graph_emb, graph_emb_enhanced])
  ```


  Option B: Residual Fusion
  ─────────────────────────
  Add text as residual to graph (graph is primary):

  ```python
  # Project text to same dim as graph
  text_residual = Dense(graph_dim)(text_emb)

  # Scale down text contribution
  alpha = 0.1  # Small weight for text
  fused = graph_emb + alpha * text_residual
  ```


  Option C: Contrastive Learning
  ───────────────────────────────
  Instead of direct fusion, use contrastive loss to align modalities:

  ```python
  # Contrastive loss: maximize similarity of graph-text pairs for same sample
  graph_norm = tf.nn.l2_normalize(graph_emb, axis=-1)
  text_norm = tf.nn.l2_normalize(text_emb, axis=-1)

  similarity = tf.matmul(graph_norm, text_norm, transpose_b=True)
  labels_matrix = tf.eye(batch_size)

  contrastive_loss = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels_matrix,
      logits=similarity / temperature
  )
  ```


  Option D: No Fusion - Separate Predictions
  ───────────────────────────────────────────
  Train graph and text separately, ensemble predictions:

  ```python
  graph_pred = graph_head(graph_emb)
  text_pred = text_head(text_emb)

  # Learned weighted average
  weight = Sigmoid()(Dense(1)(Concatenate()([graph_emb, text_emb])))
  final_pred = weight * graph_pred + (1 - weight) * text_pred
  ```
""")

    print("="*80 + "\n")


def provide_quick_fixes_to_try():
    """Provide quick fixes to try immediately."""

    print("\n" + "="*80)
    print("QUICK FIXES TO TRY NOW (30 minutes each)")
    print("="*80)

    print("""
FIX #1: DISABLE MIDDLE FUSION
──────────────────────────────

Rationale: Middle fusion may corrupt graph representations.

Implementation:
  In your training config or code:
  ```python
  model = make_model_multimodal_v5(
      use_middle_fusion=False,  # ← Change this
      late_fusion_type='gated',
      ...
  )
  ```

Expected improvement: +0.5 to 1.0 MAE
Time: 30 minutes to retrain


FIX #2: HEAVY TEXT REGULARIZATION
──────────────────────────────────

Rationale: Force model to rely more on graph by heavily regularizing text.

Implementation:
  Edit kgcnn/literature/DenseGNN/_make_dense_multimodal_v5.py, line ~129:

  ```python
  text_projection = ProjectionHead(
      embedding_dim=text_embedding_dim,
      projection_dim=text_projection_dim,
      dropout=0.5  # ← Change from 0.1 to 0.5
  )
  text_emb = text_projection(text_input)

  # Add additional dropout
  text_emb = tf.keras.layers.Dropout(0.3)(text_emb)  # ← Add this line
  ```

Expected improvement: +0.3 to 0.8 MAE
Time: 30 minutes to retrain


FIX #3: REDUCE TEXT INFLUENCE VIA GATE INITIALIZATION
──────────────────────────────────────────────────────

Rationale: Initialize fusion gate to favor graph (0.8 = 80% graph, 20% text).

Implementation:
  Edit kgcnn/literature/DenseGNN/_multimodal_fusion.py, in GatedFusion class:

  ```python
  class GatedFusion(tf.keras.layers.Layer):
      def __init__(self, ...):
          super().__init__()
          ...
          # Change gate initialization
          self.gate = self.add_weight(
              name='gate',
              shape=(1,),
              initializer=tf.keras.initializers.Constant(0.8),  # ← Change from 0.5
              trainable=True
          )
  ```

Expected improvement: +0.2 to 0.5 MAE
Time: 30 minutes to retrain


FIX #4: INCREASE GRAPH CAPACITY, DECREASE TEXT
───────────────────────────────────────────────

Rationale: Give graph more capacity to compete with text.

Implementation:
  In your config:
  ```python
  model = make_model_multimodal_v5(
      graph_projection_dim=256,  # ← Increase from 128
      text_projection_dim=64,    # ← Decrease from 128
      ...
  )
  ```

Expected improvement: +0.3 to 0.7 MAE
Time: 30 minutes to retrain


FIX #5: REMOVE TEXT ENTIRELY (SANITY CHECK)
────────────────────────────────────────────

Rationale: Confirm that pure graph is indeed better.

Implementation:
  Use the baseline DenseGNN Lite model (no text).

Expected result: Should match your baseline (18.79 MAE)
Time: Already done - you have this result


FIX #6: TEXT-ONLY BASELINE (DIAGNOSTIC)
────────────────────────────────────────

Rationale: Check if text alone is useful.

Implementation:
  ```python
  # Simple text-only model
  text_input = tf.keras.Input(shape=(768,))
  x = tf.keras.layers.Dense(128, activation='relu')(text_input)
  x = tf.keras.layers.Dropout(0.3)(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  output = tf.keras.layers.Dense(1)(x)

  model = tf.keras.Model(text_input, output)
  ```

Expected result:
  - If text-only MAE < 25: Text is useful, problem is in fusion
  - If text-only MAE > 40: Text is not useful for this task
Time: 20 minutes to train
""")

    print("="*80 + "\n")


def recommend_best_strategy():
    """Recommend the best strategy based on current results."""

    print("\n" + "="*80)
    print("RECOMMENDED STRATEGY (Based on Your Results)")
    print("="*80)

    print("""
Based on:
  - DenseGNN Baseline: 18.79 MAE
  - Multimodal (LayerNorm): 20.06 MAE
  - Multimodal is 6.8% worse

My hypothesis (80% confident):
  🔴 **TEXT IS NOT USEFUL FOR YOUR PREDICTION TASK**

Reasons:
  1. Even with balanced scales (LayerNorm), multimodal is worse
  2. The gap (1.27 MAE) is significant (6.8%)
  3. You're likely predicting a graph-structural property (band gap? formation energy?)
     that text descriptions don't capture well

Recommended action plan:
─────────────────────────

PHASE 1: DIAGNOSE TEXT QUALITY (1-2 hours)
───────────────────────────────────────────

  1. Train text-only baseline:
     ```bash
     python train_text_only.py
     ```

     If text-only MAE > 40:
       → **TEXT IS USELESS - REMOVE IT**
       → Use pure graph model (your baseline)
       → Focus on improving graph architecture instead

     If text-only MAE < 30:
       → Text contains useful signal
       → Problem is in fusion
       → Continue to Phase 2

  2. Check text data manually:
     ```bash
     python -c "
     import pickle
     with open('data/dataset.pkl', 'rb') as f:
         data = pickle.load(f)

     # Print 10 examples
     for i in range(10):
         print(f'Sample {i}:')
         print(f'  Text: {data[i][\"text\"]}')
         print(f'  Label: {data[i][\"label\"]}')
         print()
     "
     ```

     Look for:
       ✓ Is text relevant to prediction target?
       ✓ Is text diverse across samples?
       ✗ Does text leak target information?


PHASE 2: IF TEXT IS USEFUL, FIX FUSION (2-3 hours)
───────────────────────────────────────────────────

  Try these in order:

  1. **Disable middle fusion** (easiest)
     Expected improvement: +0.5 MAE

  2. **Heavy text dropout** (0.5)
     Expected improvement: +0.4 MAE

  3. **Increase graph capacity** (graph_dim=256, text_dim=64)
     Expected improvement: +0.3 MAE

  4. **Try residual fusion** (graph + 0.1 * text)
     Expected improvement: +0.6 MAE

  5. **Try attention fusion**
     Expected improvement: +0.8 MAE (but more complex)


PHASE 3: IF NOTHING WORKS, ABANDON TEXT (30 minutes)
─────────────────────────────────────────────────────

  If after trying all fixes, multimodal is still worse than baseline:

  → **Your text data is not suitable for this task**

  Options:
    A. Use pure graph model (current baseline)
    B. Get better text data (human-written, task-specific)
    C. Generate text differently (focus on structural properties)
    D. Use different text source (literature abstracts, papers)


IMMEDIATE NEXT STEP (RIGHT NOW - 5 minutes):
─────────────────────────────────────────────

Run text-only baseline to check if text is even useful:

```bash
cd /home/user/DenseGNN

# Quick test
python -c "
import tensorflow as tf
import numpy as np
# Load your data
# ... load text_embeddings and labels ...

# Train simple model
text_input = tf.keras.Input(shape=(768,))
x = tf.keras.layers.Dense(128, activation='relu')(text_input)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(text_input, output)

model.compile(optimizer='adam', loss='mae', metrics=['mae'])
model.fit(text_train, y_train, validation_split=0.2, epochs=50)

# Check test MAE
test_mae = model.evaluate(text_test, y_test)[1]
print(f'Text-only MAE: {test_mae:.2f}')

if test_mae > 40:
    print('🔴 TEXT IS NOT USEFUL - REMOVE IT!')
elif test_mae < 25:
    print('✅ TEXT IS USEFUL - PROBLEM IS IN FUSION')
else:
    print('⚠️ TEXT IS MARGINALLY USEFUL - QUESTIONABLE')
"
```

This will immediately tell you if pursuing multimodal is worth it!
""")

    print("="*80 + "\n")


if __name__ == '__main__':
    analyze_why_layernorm_didnt_help()
    provide_next_diagnostic_steps()
    provide_quick_fixes_to_try()
    recommend_best_strategy()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Your current situation:
  - Baseline (graph-only): 18.79 MAE
  - Multimodal (with LayerNorm): 20.06 MAE  ← 6.8% worse!

Most likely root cause:
  🔴 TEXT IS NOT USEFUL FOR YOUR PREDICTION TASK

Next immediate action:
  1. Train text-only baseline (5 minutes)
  2. If text-only MAE > 40: Remove text, use graph-only
  3. If text-only MAE < 25: Problem is fusion, try fixes above

The diagnostic tools are ready. Run them to find the exact problem!
""")
    print("="*80 + "\n")
