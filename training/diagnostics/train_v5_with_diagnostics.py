"""
Simplified training script for v5 with diagnostics

This shows the EASIEST way to add diagnostics: manually compute embeddings
in your training loop and log them.

This approach works with your existing model without modification!
"""

import os
import sys
import tensorflow as tf
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.diagnostics.multimodal_diagnostics import MultimodalDiagnostics
from kgcnn.literature.DenseGNN import make_model_multimodal_v5


def build_embedding_extractors(model_config):
    """
    Build separate "submodels" for graph and text pathways.

    This lets us extract embeddings without modifying the main model.

    Args:
        model_config: Same config used to build main model

    Returns:
        tuple: (graph_model, text_model) that output embeddings
    """
    # This is a simplified example - adjust based on your actual model structure

    # Text embedding extractor
    from kgcnn.literature.DenseGNN._multimodal_fusion import ProjectionHead

    text_input = tf.keras.Input(shape=(768,), name='text_embedding')
    text_projection = ProjectionHead(
        embedding_dim=768,
        projection_dim=128,
        dropout=0.1
    )
    text_emb = text_projection(text_input)

    text_model = tf.keras.Model(inputs=text_input, outputs=text_emb, name='text_extractor')

    # For graph embedding, we'd need to build the graph network up to projection
    # This is more complex - for now, return None and extract differently

    return None, text_model


def compute_graph_embedding_manually(model, batch, config):
    """
    Manually compute graph embedding by reconstructing the forward pass.

    This is a workaround for extracting intermediate embeddings.

    Args:
        model: Your trained model
        batch: Input batch
        config: Model configuration

    Returns:
        graph_emb: Graph embeddings before fusion
    """
    # This is a placeholder - you'd need to implement based on your exact model structure

    # Option 1: Build a submodel that ends at graph_projection
    # Option 2: Use gradient tape to trace intermediate activations
    # Option 3: Modify model to output embeddings (see v5_diagnostic.py)

    # For now, return None - user should implement
    return None


def train_with_manual_logging():
    """
    Training loop with manual embedding extraction and logging.

    This is the PRACTICAL approach that works with existing models.
    """

    # ========== Setup ==========
    print("Setting up training with diagnostics...")

    # Initialize diagnostics
    diagnostics = MultimodalDiagnostics(
        log_dir='./logs/v5_manual_diagnostics',
        enable_tb=True
    )

    # Build model
    # ... your model building code here ...
    # model = make_model_multimodal_v5(...)

    # ========== Training Loop ==========
    print("Starting training...")

    for epoch in range(100):  # num_epochs
        epoch_loss = []

        for batch_idx, batch in enumerate(train_dataset):

            # ===== Forward pass with gradient tape =====
            with tf.GradientTape() as tape:
                predictions = model(batch, training=True)
                loss = loss_fn(batch['labels'], predictions)

            # ===== Backward pass =====
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_loss.append(float(loss))

            # ===== Diagnostic logging every N batches =====
            if batch_idx % 20 == 0:  # Log every 20 batches

                # Option 1: Extract embeddings using submodels
                text_input = batch['text_embedding'] if isinstance(batch, dict) else batch[-1]
                text_emb = text_extractor(text_input, training=False)  # Separate text model
                graph_emb = compute_graph_embedding_manually(model, batch, config)

                # Option 2: Extract from model layers (if you know layer names)
                # text_layer = model.get_layer('text_projection')
                # text_emb = text_layer.output

                # Log if we successfully extracted embeddings
                if graph_emb is not None and text_emb is not None:
                    diagnostics.log_embeddings(
                        graph_emb,
                        text_emb,
                        epoch=epoch,
                        batch_idx=batch_idx
                    )

                # Log current loss
                diagnostics.log_loss(loss, epoch=epoch, with_text=True)

        # ===== End of epoch =====
        avg_loss = np.mean(epoch_loss)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        # Extract and log gate weights
        # You need to know which layer has the gate
        try:
            # Find GatedFusion layer
            for layer in model.layers:
                if 'gated_fusion' in layer.name.lower() or 'GatedFusion' in str(type(layer)):
                    if hasattr(layer, 'gate'):
                        gate = layer.gate.numpy()
                        diagnostics.log_fusion_gates(gate, epoch=epoch)
                        print(f"  Gate weight: {np.mean(gate):.3f}")
                        break
        except Exception as e:
            print(f"  Warning: Could not extract gate weights: {e}")

    # ===== Generate final report =====
    print("\n\nGenerating diagnostic report...")
    report = diagnostics.generate_report(save_plots=True)

    return model, report


# ============================================================================
# EASIEST APPROACH: Extract embeddings by modifying model architecture
# ============================================================================

def make_v5_with_exposed_embeddings(config):
    """
    Build v5 model with embeddings exposed as additional outputs.

    This is the EASIEST and most reliable approach!

    Returns:
        model: Multi-output model that returns (prediction, graph_emb, text_emb)
    """
    from kgcnn.literature.DenseGNN._make_dense_multimodal_v5 import (
        make_model_multimodal_v5, get_features, GraphNetworkConfigurator
    )
    from kgcnn.layers.geom import EuclideanNorm
    from kgcnn.literature.DenseGNN._multimodal_fusion import ProjectionHead, GatedFusion
    from kgcnn.layers.pooling import PoolingGlobalEdges
    from kgcnn.layers.mlp import MLP, GraphMLP
    from kgcnn.layers.modules import LazyConcatenate
    from kgcnn.literature.DenseGNN._gin_conv import GINELITE

    # Copy the model building code from _make_dense_multimodal_v5.py
    # but modify the final return to output multiple values

    # ... (copy model building code) ...

    # Instead of:
    # return ks.Model(inputs=input_list, outputs=out, name=name)

    # Do:
    # return ks.Model(inputs=input_list, outputs=[out, graph_emb, text_emb], name=name)

    # This way you can do:
    # pred, graph_emb, text_emb = model(batch)

    # For now, this is a placeholder - user should modify their model file

    raise NotImplementedError(
        "Modify your make_model_multimodal_v5() to return [out, graph_emb, text_emb]"
    )


def train_with_multi_output_model():
    """
    Training with a multi-output model (easiest approach).

    Modify your model to output [prediction, graph_emb, text_emb] instead of just prediction.
    """

    diagnostics = MultimodalDiagnostics(log_dir='./logs/v5_multi_output')

    # Build multi-output model
    # model = make_v5_with_exposed_embeddings(config)

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataset):

            with tf.GradientTape() as tape:
                # Get all outputs
                predictions, graph_emb, text_emb = model(batch, training=True)

                loss = loss_fn(batch['labels'], predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Diagnostic logging is now trivial!
            if batch_idx % 20 == 0:
                diagnostics.log_embeddings(graph_emb, text_emb, epoch, batch_idx)
                diagnostics.log_loss(loss, epoch, with_text=True)

        # End of epoch
        # Extract gate (from model layer)
        gate = model.get_layer('gated_fusion').gate.numpy()
        diagnostics.log_fusion_gates(gate, epoch)

    report = diagnostics.generate_report(save_plots=True)
    return report


# ============================================================================
# RECOMMENDED: Patch to add to your existing training script
# ============================================================================

"""
RECOMMENDED APPROACH: Add this code to your existing training script

# At the top of your training script:
from training.diagnostics.multimodal_diagnostics import MultimodalDiagnostics, quick_diagnostic_check

# Before training loop:
diagnostics = MultimodalDiagnostics(log_dir='./logs/diagnostics')

# Modify your model building to return embeddings:
# In kgcnn/literature/DenseGNN/_make_dense_multimodal_v5.py, line 198:
# Change:
#   return ks.Model(inputs=input_list, outputs=out, name=name)
# To:
#   return ks.Model(inputs=input_list, outputs={'prediction': out, 'graph_emb': graph_emb, 'text_emb': text_emb}, name=name)

# In your training loop:
with tf.GradientTape() as tape:
    outputs = model(batch, training=True)

    # Extract outputs
    if isinstance(outputs, dict):
        predictions = outputs['prediction']
        graph_emb = outputs.get('graph_emb')
        text_emb = outputs.get('text_emb')
    else:
        predictions = outputs
        graph_emb = None
        text_emb = None

    loss = loss_fn(labels, predictions)

# ... gradient update ...

# Log diagnostics every N batches
if batch_idx % 20 == 0 and graph_emb is not None and text_emb is not None:
    diagnostics.log_embeddings(graph_emb, text_emb, epoch, batch_idx)
    diagnostics.log_loss(loss, epoch, with_text=True)

# End of epoch: log gate weights
try:
    gate_layer = model.get_layer('gated_fusion')
    diagnostics.log_fusion_gates(gate_layer.gate.numpy(), epoch)
except:
    pass

# After training:
diagnostics.generate_report(save_plots=True)
"""


def print_integration_instructions():
    """Print step-by-step instructions for adding diagnostics."""

    print("\n" + "=" * 80)
    print("HOW TO ADD DIAGNOSTICS TO YOUR TRAINING")
    print("=" * 80)

    print("\n📝 STEP 1: Modify your model to expose embeddings")
    print("-" * 80)
    print("""
Edit: kgcnn/literature/DenseGNN/_make_dense_multimodal_v5.py

At line 198, change the return statement from:
    return ks.Model(inputs=input_list, outputs=out, name=name)

To:
    return ks.Model(
        inputs=input_list,
        outputs={'prediction': out, 'graph_emb': graph_emb, 'text_emb': text_emb},
        name=name
    )

This makes the model output embeddings along with predictions.
""")

    print("\n📝 STEP 2: Add diagnostics to your training script")
    print("-" * 80)
    print("""
At the top of your training script, add:

    from training.diagnostics.multimodal_diagnostics import MultimodalDiagnostics

Before your training loop:

    diagnostics = MultimodalDiagnostics(log_dir='./logs/v5_diagnostics', enable_tb=True)
""")

    print("\n📝 STEP 3: Extract outputs in training loop")
    print("-" * 80)
    print("""
In your training loop, change:

    predictions = model(batch, training=True)

To:

    outputs = model(batch, training=True)
    predictions = outputs['prediction']
    graph_emb = outputs['graph_emb']
    text_emb = outputs['text_emb']
""")

    print("\n📝 STEP 4: Log embeddings")
    print("-" * 80)
    print("""
Add logging after your gradient update:

    if batch_idx % 20 == 0:  # Log every 20 batches
        diagnostics.log_embeddings(graph_emb, text_emb, epoch, batch_idx)
        diagnostics.log_loss(loss, epoch, with_text=True)
""")

    print("\n📝 STEP 5: Log gate weights (end of epoch)")
    print("-" * 80)
    print("""
At the end of each epoch:

    gate_layer = model.get_layer('gated_fusion')
    diagnostics.log_fusion_gates(gate_layer.gate.numpy(), epoch)
""")

    print("\n📝 STEP 6: Generate report (end of training)")
    print("-" * 80)
    print("""
After training completes:

    print("\\nGenerating diagnostic report...")
    report = diagnostics.generate_report(save_plots=True)
    print(f"Report saved to: {diagnostics.log_dir}/diagnostic_report.json")
""")

    print("\n📊 STEP 7: View results")
    print("-" * 80)
    print("""
1. Check the printed diagnostic report
2. View plots in: ./logs/v5_diagnostics/plots/
3. Launch TensorBoard: tensorboard --logdir ./logs/v5_diagnostics/tensorboard
4. Open: http://localhost:6006
""")

    print("\n" + "=" * 80)
    print("That's it! You'll now have comprehensive diagnostics during training.")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    print_integration_instructions()
