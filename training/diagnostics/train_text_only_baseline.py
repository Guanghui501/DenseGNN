"""
Train Text-Only Baseline

This script trains a simple MLP on BERT text embeddings only.
It answers the critical question: "Is text actually useful for prediction?"

Usage:
    python train_text_only_baseline.py --data_config ../hyper/hyper_multimodal_v5.py

Result interpretation:
    - Text-only MAE < 25:  Text is useful, problem is in fusion
    - Text-only MAE 25-40: Text is marginally useful
    - Text-only MAE > 40:  Text is NOT useful, remove it!
"""

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def build_text_only_model(text_dim=768, hidden_dims=[128, 64], dropout=0.3):
    """
    Build simple MLP for text-only prediction.

    Args:
        text_dim: BERT embedding dimension
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate

    Returns:
        keras.Model
    """
    text_input = keras.Input(shape=(text_dim,), name='text_embedding')

    x = text_input
    for i, dim in enumerate(hidden_dims):
        x = keras.layers.Dense(dim, activation='relu', name=f'hidden_{i}')(x)
        x = keras.layers.Dropout(dropout, name=f'dropout_{i}')(x)

    output = keras.layers.Dense(1, name='output')(x)

    model = keras.Model(inputs=text_input, outputs=output, name='text_only_baseline')

    return model


def train_text_only(dataset, text_dim=768, epochs=100, batch_size=32):
    """
    Train text-only baseline model.

    Args:
        dataset: Dataset with text embeddings and labels
        text_dim: BERT embedding dimension
        epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        tuple: (model, history, test_mae)
    """
    print("\n" + "="*80)
    print("TRAINING TEXT-ONLY BASELINE")
    print("="*80)

    # Build model
    model = build_text_only_model(text_dim=text_dim)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mae',
        metrics=['mae']
    )

    print("\nModel architecture:")
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
    ]

    # Train
    print("\nTraining...")
    history = model.fit(
        dataset['train'],
        validation_data=dataset['val'],
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\nEvaluating on test set...")
    test_results = model.evaluate(dataset['test'], verbose=0)
    test_mae = test_results[1]  # MAE metric

    print(f"\n{'='*80}")
    print("RESULTS")
    print("="*80)
    print(f"Best validation MAE: {min(history.history['val_mae']):.2f}")
    print(f"Test MAE:            {test_mae:.2f}")
    print("="*80)

    return model, history, test_mae


def interpret_results(test_mae, baseline_graph_mae=18.79):
    """
    Interpret text-only results and provide recommendations.

    Args:
        test_mae: Text-only model test MAE
        baseline_graph_mae: Graph-only baseline MAE for comparison
    """
    print("\n" + "="*80)
    print("INTERPRETATION & RECOMMENDATIONS")
    print("="*80)

    ratio = test_mae / baseline_graph_mae

    print(f"\nComparison:")
    print(f"  Graph-only baseline: {baseline_graph_mae:.2f} MAE")
    print(f"  Text-only baseline:  {test_mae:.2f} MAE")
    print(f"  Ratio (Text/Graph):  {ratio:.2f}x")

    if test_mae > baseline_graph_mae * 2.5:
        # Text is very poor
        print(f"\n🔴 VERDICT: TEXT IS NOT USEFUL FOR THIS TASK")
        print(f"\n   Text-only performance ({test_mae:.2f}) is {ratio:.1f}x worse than graph ({baseline_graph_mae:.2f})")
        print("   Text does not contain predictive information for your target property.")
        print("\n   RECOMMENDATION: 🚫 REMOVE TEXT FROM YOUR MODEL")
        print("   - Use pure graph baseline (DenseGNN Lite)")
        print("   - Focus on improving graph architecture instead")
        print("   - Consider using different/better text if available")

    elif test_mae > baseline_graph_mae * 1.5:
        # Text is poor but has some signal
        print(f"\n⚠️  VERDICT: TEXT IS MARGINALLY USEFUL")
        print(f"\n   Text-only performance ({test_mae:.2f}) is {ratio:.1f}x worse than graph ({baseline_graph_mae:.2f})")
        print("   Text contains weak signal but graph is much better.")
        print("\n   RECOMMENDATION: 🟡 TEXT IS QUESTIONABLE")
        print("   - Try multimodal with HEAVY text regularization (dropout=0.5)")
        print("   - Use late fusion only (no middle fusion)")
        print("   - Initialize gate to favor graph (0.9 = 90% graph, 10% text)")
        print("   - If multimodal still worse than baseline, remove text")

    elif test_mae > baseline_graph_mae:
        # Text is weaker than graph but useful
        print(f"\n✅ VERDICT: TEXT IS USEFUL BUT WEAKER THAN GRAPH")
        print(f"\n   Text-only performance ({test_mae:.2f}) is {ratio:.1f}x worse than graph ({baseline_graph_mae:.2f})")
        print("   Text contains useful information, but graph is primary modality.")
        print("\n   RECOMMENDATION: 💡 MULTIMODAL CAN WORK")
        print("   - Text should complement graph, not replace it")
        print("   - Use asymmetric fusion:")
        print("     * graph_projection_dim = 256 (large)")
        print("     * text_projection_dim = 64 (small)")
        print("   - Add text dropout (0.3-0.5)")
        print("   - Try residual fusion: graph_emb + 0.2 * text_emb")

    elif test_mae < baseline_graph_mae * 0.8:
        # Text is better than graph!
        print(f"\n✅ VERDICT: TEXT IS VERY USEFUL (Better than graph!)")
        print(f"\n   Text-only performance ({test_mae:.2f}) is BETTER than graph ({baseline_graph_mae:.2f})")
        print("   Text contains strong predictive signal.")
        print("\n   RECOMMENDATION: 💡 MULTIMODAL SHOULD DEFINITELY WORK")
        print("   - Your current multimodal worse than both modalities suggests:")
        print("     * Fusion mechanism is broken")
        print("     * Negative transfer between modalities")
        print("   - Try these fixes:")
        print("     * Use attention-based fusion instead of gating")
        print("     * Train with contrastive loss to align modalities")
        print("     * Use separate prediction heads with ensemble")

    else:
        # Text is comparable to graph
        print(f"\n✅ VERDICT: TEXT IS AS GOOD AS GRAPH")
        print(f"\n   Text-only ({test_mae:.2f}) ≈ Graph-only ({baseline_graph_mae:.2f})")
        print("   Both modalities have similar predictive power.")
        print("\n   RECOMMENDATION: 💡 MULTIMODAL SHOULD HELP")
        print("   - Use balanced fusion (equal projection dims)")
        print("   - Try attention to capture complementary information")
        print("   - Current poor multimodal performance suggests:")
        print("     * Negative transfer or interference")
        print("     * Optimization issues (different modalities need different LR)")

    print("\n" + "="*80 + "\n")


def plot_training_curves(history, save_path='./text_only_training.png'):
    """Plot training curves."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(history.history['loss'], label='Train Loss', alpha=0.7)
    ax1.plot(history.history['val_loss'], label='Val Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MAE Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MAE
    ax2.plot(history.history['mae'], label='Train MAE', alpha=0.7)
    ax2.plot(history.history['val_mae'], label='Val MAE', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title('Training MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train text-only baseline')
    parser.add_argument('--data_config', type=str, help='Path to data config file')
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--text_dim', type=int, default=768, help='Text embedding dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--graph_baseline_mae', type=float, default=18.79,
                       help='Graph baseline MAE for comparison')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("TEXT-ONLY BASELINE EXPERIMENT")
    print("="*80)
    print("\nThis experiment tests if text contains useful information for prediction.")
    print("We train a simple MLP on BERT embeddings and compare to graph baseline.")
    print("\nInterpretation:")
    print("  - If text-only MAE > 2.5x graph baseline → TEXT IS USELESS")
    print("  - If text-only MAE ≈ graph baseline → TEXT IS USEFUL")
    print("  - If text-only MAE < graph baseline → TEXT IS VERY USEFUL")
    print("="*80)

    # Load data
    print("\n[1] Loading data...")

    if args.data_path:
        # Load from file
        print(f"Loading from: {args.data_path}")
        # TODO: Implement data loading based on your format
        # dataset = load_dataset(args.data_path)
        raise NotImplementedError("Please implement data loading for your format")

    elif args.data_config:
        # Load using config
        print(f"Loading using config: {args.data_config}")
        # TODO: Implement config-based loading
        raise NotImplementedError("Please implement config-based data loading")

    else:
        print("ERROR: Please provide either --data_path or --data_config")
        print("\nExample usage:")
        print("  python train_text_only_baseline.py --data_path ./data/matbench.pkl")
        return

    # Train
    print("\n[2] Training text-only model...")
    model, history, test_mae = train_text_only(
        dataset,
        text_dim=args.text_dim,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Plot
    print("\n[3] Generating plots...")
    plot_training_curves(history)

    # Interpret
    print("\n[4] Interpreting results...")
    interpret_results(test_mae, args.graph_baseline_mae)

    # Save model
    model_path = './text_only_baseline.h5'
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    print("\n✅ Text-only baseline experiment complete!")


def quick_test_with_dummy_data():
    """Quick test with dummy data to demonstrate functionality."""

    print("\n" + "="*80)
    print("QUICK TEST WITH DUMMY DATA")
    print("="*80)
    print("(For demonstration - replace with real data)")

    # Create dummy data
    n_train, n_val, n_test = 1000, 200, 200
    text_dim = 768

    # Simulate text embeddings with weak signal
    # Real text should have structure, this is random + small signal
    np.random.seed(42)

    def create_dummy_data(n_samples):
        # Random text embeddings
        text_emb = np.random.randn(n_samples, text_dim).astype(np.float32)

        # Labels with weak correlation to text (simulates poor text quality)
        # Real correlation should come from actual text semantics
        text_signal = np.sum(text_emb[:, :10], axis=1)  # Use first 10 dims
        noise = np.random.randn(n_samples) * 20
        labels = text_signal + noise + 18.0  # Center around 18 (similar to your MAE)
        labels = labels.reshape(-1, 1).astype(np.float32)

        return text_emb, labels

    text_train, y_train = create_dummy_data(n_train)
    text_val, y_val = create_dummy_data(n_val)
    text_test, y_test = create_dummy_data(n_test)

    # Create tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((text_train, y_train)).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((text_val, y_val)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((text_test, y_test)).batch(32)

    dataset = {'train': train_ds, 'val': val_ds, 'test': test_ds}

    # Train
    model, history, test_mae = train_text_only(dataset, text_dim=text_dim, epochs=50)

    # Interpret
    interpret_results(test_mae, baseline_graph_mae=18.79)

    print("\n💡 This was a test with dummy data.")
    print("   Replace with your actual dataset to get real results!")


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        # No arguments - run quick test
        print("No arguments provided. Running quick test with dummy data...\n")
        quick_test_with_dummy_data()
    else:
        # Run with real data
        main()
