"""
Example: How to integrate MultimodalDiagnostics into v5 training

This shows where to add diagnostic logging calls in your training loop.
"""

import tensorflow as tf
from training.diagnostics.multimodal_diagnostics import MultimodalDiagnostics, quick_diagnostic_check


# ============================================================================
# Example 1: Full diagnostic integration in training loop
# ============================================================================

def train_with_diagnostics():
    """Example training loop with full diagnostics."""

    # Initialize diagnostics
    diagnostics = MultimodalDiagnostics(
        log_dir='./logs/v5_diagnostics',
        enable_tb=True
    )

    # ... your model initialization code ...
    # model = make_model_multimodal_v5(...)

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataset):

            # Training step
            with tf.GradientTape() as tape:
                # Forward pass - need to capture intermediate embeddings
                graph_emb = model.get_graph_embedding(batch)  # Extract before fusion
                text_emb = model.get_text_embedding(batch)    # Extract text embedding

                # Log embeddings every N batches
                if batch_idx % 10 == 0:  # Log every 10 batches
                    diagnostics.log_embeddings(
                        graph_emb, text_emb,
                        epoch=epoch,
                        batch_idx=batch_idx
                    )

                # Get final prediction
                predictions = model(batch, training=True)
                loss = loss_fn(batch['labels'], predictions)

            # Backward pass
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Log loss
            if batch_idx % 10 == 0:
                diagnostics.log_loss(loss, epoch=epoch, with_text=True)

        # End of epoch: log fusion gate weights if available
        if hasattr(model, 'late_fusion') and hasattr(model.late_fusion, 'gate'):
            gate_weights = model.late_fusion.gate.numpy()
            diagnostics.log_fusion_gates(gate_weights, epoch=epoch)

        print(f"Epoch {epoch} completed")

    # Generate final report
    print("\n\nGenerating diagnostic report...")
    report = diagnostics.generate_report(save_plots=True)

    return report


# ============================================================================
# Example 2: Quick diagnostic check (no logging infrastructure)
# ============================================================================

def quick_check_example():
    """Quick one-time diagnostic check."""

    # Assume you have a trained model
    # model = load_model(...)

    # Get embeddings from a batch
    batch = next(iter(test_dataset))

    graph_emb = model.get_graph_embedding(batch)
    text_emb = model.get_text_embedding(batch)

    # Quick check
    results = quick_diagnostic_check(graph_emb, text_emb)

    # Based on results, apply fixes
    if results['imbalanced'] and results['std_ratio'] > 3.0:
        print("\n💡 Suggested fix: Scale down graph embeddings")
        scale_factor = 1.0 / results['std_ratio']
        print(f"   Try: graph_emb = graph_emb * {scale_factor:.3f}")


# ============================================================================
# Example 3: Modify model to expose intermediate embeddings
# ============================================================================

# You need to modify your model to expose embeddings BEFORE fusion.
# Add these methods to your model class:

"""
class DenseGNNMultimodalV5(tf.keras.Model):

    def get_graph_embedding(self, inputs):
        '''Extract graph embedding before fusion.'''
        # ... run graph network ...
        # out = graph_network(...)
        # return out  # Shape: (batch, graph_dim)
        pass

    def get_text_embedding(self, inputs):
        '''Extract text embedding.'''
        # text_emb = self.text_encoder(inputs['text'])
        # return text_emb  # Shape: (batch, text_dim)
        pass
"""


# ============================================================================
# Example 4: Ablation study - train with and without text
# ============================================================================

def ablation_study():
    """Compare training with and without text."""

    diagnostics = MultimodalDiagnostics(log_dir='./logs/ablation_study')

    # Train WITHOUT text (pure graph)
    print("Training without text...")
    model_no_text = train_model(use_text=False)
    for epoch, loss in enumerate(model_no_text.history['loss']):
        diagnostics.log_loss(loss, epoch=epoch, with_text=False)

    # Train WITH text (multimodal)
    print("Training with text...")
    model_with_text = train_model(use_text=True)
    for epoch, loss in enumerate(model_with_text.history['loss']):
        diagnostics.log_loss(loss, epoch=epoch, with_text=True)

    # Generate comparison report
    diagnostics.generate_report()


# ============================================================================
# Example 5: Custom callback for automatic logging
# ============================================================================

class DiagnosticCallback(tf.keras.callbacks.Callback):
    """Keras callback for automatic diagnostic logging."""

    def __init__(self, diagnostics, log_frequency=10):
        super().__init__()
        self.diagnostics = diagnostics
        self.log_frequency = log_frequency
        self.batch_count = 0

    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1

        # Log every N batches
        if self.batch_count % self.log_frequency == 0:
            # Extract embeddings from model
            if hasattr(self.model, 'graph_embedding'):
                graph_emb = self.model.graph_embedding
                text_emb = self.model.text_embedding

                self.diagnostics.log_embeddings(
                    graph_emb, text_emb,
                    epoch=self.model.epoch_counter,
                    batch_idx=batch
                )

    def on_epoch_end(self, epoch, logs=None):
        # Log gate weights
        if hasattr(self.model, 'late_fusion'):
            gate = self.model.late_fusion.gate
            self.diagnostics.log_fusion_gates(gate, epoch=epoch)

        # Log loss
        if logs and 'loss' in logs:
            self.diagnostics.log_loss(logs['loss'], epoch=epoch, with_text=True)

    def on_train_end(self, logs=None):
        # Generate final report
        print("\n\nGenerating diagnostic report...")
        self.diagnostics.generate_report(save_plots=True)


# Usage:
# diagnostics = MultimodalDiagnostics(log_dir='./logs/v5_diagnostics')
# callback = DiagnosticCallback(diagnostics, log_frequency=10)
# model.fit(train_dataset, epochs=100, callbacks=[callback])


# ============================================================================
# Example 6: Post-training analysis from saved embeddings
# ============================================================================

def analyze_saved_embeddings():
    """Analyze embeddings saved during training."""

    import numpy as np

    # Load saved embeddings
    graph_embeddings = np.load('embeddings/graph_emb.npy')
    text_embeddings = np.load('embeddings/text_emb.npy')

    diagnostics = MultimodalDiagnostics(log_dir='./logs/post_analysis')

    # Log all embeddings
    for i, (g_emb, t_emb) in enumerate(zip(graph_embeddings, text_embeddings)):
        diagnostics.log_embeddings(g_emb, t_emb, epoch=i)

    # Generate report
    report = diagnostics.generate_report(save_plots=True)

    return report


if __name__ == '__main__':
    print("This is an example file showing how to integrate diagnostics.")
    print("Copy the relevant code into your actual training script.")
    print("\nKey integration points:")
    print("1. Initialize MultimodalDiagnostics at start of training")
    print("2. Call log_embeddings() in training loop")
    print("3. Call log_fusion_gates() after each epoch")
    print("4. Call generate_report() at end of training")
