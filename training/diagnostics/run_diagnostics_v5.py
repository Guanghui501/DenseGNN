"""
Ready-to-run diagnostic script for DenseGNN v5 multimodal

This script loads your trained model and runs comprehensive diagnostics.

Usage:
    python run_diagnostics_v5.py --model_path ./models/v5_model.h5 \\
                                   --data_path ./data/test_dataset.pkl \\
                                   --output_dir ./diagnostic_results
"""

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics.multimodal_diagnostics import MultimodalDiagnostics, quick_diagnostic_check


class EmbeddingExtractor(tf.keras.Model):
    """
    Wrapper to extract intermediate embeddings from DenseGNN v5 model.

    This wraps your existing model and intercepts the embeddings before fusion.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # Cache for embeddings
        self.graph_emb = None
        self.text_emb = None
        self.fused_emb = None

    def call(self, inputs, training=False):
        """Forward pass with embedding extraction."""

        # We need to trace through the model to extract embeddings
        # This requires modifying the model architecture slightly

        # Assuming v5 has this structure:
        # 1. Graph network -> graph_emb
        # 2. Text encoder -> text_emb
        # 3. Fusion -> fused_emb
        # 4. Output head -> predictions

        # For now, run the full model
        output = self.base_model(inputs, training=training)

        # Try to extract embeddings from known layer names
        # (You'll need to adjust these based on your actual model)
        try:
            # Option 1: Access intermediate layers by name
            if hasattr(self.base_model, 'get_layer'):
                try:
                    graph_layer = self.base_model.get_layer('graph_output')
                    self.graph_emb = graph_layer.output
                except:
                    pass

                try:
                    text_layer = self.base_model.get_layer('text_projection')
                    self.text_emb = text_layer.output
                except:
                    pass

            # Option 2: Access from model attributes (if exposed)
            if hasattr(self.base_model, 'graph_embedding'):
                self.graph_emb = self.base_model.graph_embedding
            if hasattr(self.base_model, 'text_embedding'):
                self.text_emb = self.base_model.text_embedding

        except Exception as e:
            print(f"Warning: Could not extract embeddings: {e}")

        return output

    def get_gate_weights(self):
        """Extract fusion gate weights."""
        try:
            if hasattr(self.base_model, 'late_fusion'):
                fusion_layer = self.base_model.late_fusion
                if hasattr(fusion_layer, 'gate'):
                    return fusion_layer.gate
            # Try to find GatedFusion layer
            for layer in self.base_model.layers:
                if 'gated_fusion' in layer.name.lower():
                    if hasattr(layer, 'gate'):
                        return layer.gate
        except:
            pass
        return None


def run_diagnostics_on_dataset(model, dataset, diagnostics, num_batches=100):
    """
    Run diagnostics on a dataset.

    Args:
        model: Trained model (or EmbeddingExtractor wrapper)
        dataset: tf.data.Dataset or iterator
        diagnostics: MultimodalDiagnostics instance
        num_batches: Number of batches to process
    """

    print(f"\nProcessing {num_batches} batches for diagnostics...")

    extractor = EmbeddingExtractor(model) if not isinstance(model, EmbeddingExtractor) else model

    batch_count = 0
    for batch_idx, batch in enumerate(dataset):
        if batch_count >= num_batches:
            break

        # Forward pass
        predictions = extractor(batch, training=False)

        # Extract embeddings
        graph_emb = extractor.graph_emb
        text_emb = extractor.text_emb

        if graph_emb is not None and text_emb is not None:
            # Log embeddings
            diagnostics.log_embeddings(
                graph_emb, text_emb,
                epoch=0,
                batch_idx=batch_idx,
                step=batch_idx
            )
        else:
            # Fallback: Try to extract from batch directly if model doesn't expose embeddings
            print(f"Warning: Batch {batch_idx} - Could not extract embeddings from model")
            print("Tip: Modify your model to expose graph_embedding and text_embedding attributes")

        batch_count += 1

        if batch_count % 10 == 0:
            print(f"  Processed {batch_count}/{num_batches} batches...")

    # Extract gate weights
    gate_weights = extractor.get_gate_weights()
    if gate_weights is not None:
        diagnostics.log_fusion_gates(gate_weights, epoch=0)
    else:
        print("Warning: Could not extract fusion gate weights")

    print(f"✓ Processed {batch_count} batches")


def run_quick_diagnostic(model, dataset, num_samples=1):
    """
    Quick diagnostic check on a single batch.

    Args:
        model: Trained model
        dataset: Dataset iterator
        num_samples: Number of batches to check
    """

    print("\n" + "="*80)
    print("RUNNING QUICK DIAGNOSTIC CHECK")
    print("="*80)

    extractor = EmbeddingExtractor(model)

    for i, batch in enumerate(dataset):
        if i >= num_samples:
            break

        print(f"\nBatch {i}:")
        predictions = extractor(batch, training=False)

        graph_emb = extractor.graph_emb
        text_emb = extractor.text_emb
        gate_weights = extractor.get_gate_weights()

        if graph_emb is not None and text_emb is not None:
            results = quick_diagnostic_check(graph_emb, text_emb, gate_weights)

            # Provide recommendations
            print("\n💡 RECOMMENDATIONS:")
            if results['imbalanced']:
                if results['std_ratio'] > 3.0:
                    print(f"  1. Graph embeddings are {results['std_ratio']:.1f}x larger than text")
                    print(f"     → Add: graph_emb = graph_emb * {1.0/results['std_ratio']:.3f}")
                    print(f"     → Or: text_emb = text_emb * {results['std_ratio']:.3f}")
                else:
                    print(f"  1. Text embeddings are {1.0/results['std_ratio']:.1f}x larger than graph")
                    print(f"     → Add: text_emb = text_emb * {results['std_ratio']:.3f}")

                print("  2. Better: Add LayerNormalization before fusion")
                print("  3. Or: Use separate projection heads with normalization")

            if 'gate_weight' in results:
                if results['gate_weight'] < 0.3:
                    print("  4. Model over-relies on text - graph features being ignored")
                    print("     → Increase graph embedding dimension")
                    print("     → Add dropout to text branch (e.g., 0.3)")
                elif results['gate_weight'] > 0.7:
                    print("  4. Model ignores text - text features unhelpful")
                    print("     → Check text quality and relevance")
                    print("     → Fine-tune BERT encoder")

        else:
            print("\n❌ Could not extract embeddings from model!")
            print("\nTo enable diagnostics, modify your model to expose embeddings:")
            print("""
    # In your model's call() method, add:
    self.graph_embedding = graph_emb  # After graph network
    self.text_embedding = text_emb    # After text encoder

    # Example:
    def call(self, inputs, training=False):
        graph_emb = self.graph_network(inputs)
        text_emb = self.text_encoder(inputs['text'])

        # Expose for diagnostics
        self.graph_embedding = graph_emb
        self.text_embedding = text_emb

        # Fusion
        fused = self.fusion_layer([graph_emb, text_emb])
        output = self.output_head(fused)
        return output
            """)


def main():
    parser = argparse.ArgumentParser(description='Run diagnostics on DenseGNN v5 multimodal model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_path', type=str, default=None, help='Path to test dataset (pkl or tfrecord)')
    parser.add_argument('--output_dir', type=str, default='./diagnostic_results', help='Output directory')
    parser.add_argument('--num_batches', type=int, default=100, help='Number of batches to process')
    parser.add_argument('--quick', action='store_true', help='Run quick check only (1 batch)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        model = tf.keras.models.load_model(args.model_path, compile=False)
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nIf you have custom layers, load with custom_objects:")
        print("model = tf.keras.models.load_model(path, custom_objects={...})")
        return

    # Load dataset
    if args.data_path is None:
        print("\n⚠️  No dataset provided - using model diagnostics only")
        dataset = None
    else:
        print(f"Loading dataset from {args.data_path}...")
        try:
            if args.data_path.endswith('.pkl'):
                with open(args.data_path, 'rb') as f:
                    data = pickle.load(f)
                # Convert to tf.data.Dataset
                # (You'll need to adjust this based on your data format)
                dataset = tf.data.Dataset.from_tensor_slices(data)
                dataset = dataset.batch(args.batch_size)
            elif args.data_path.endswith('.tfrecord'):
                dataset = tf.data.TFRecordDataset(args.data_path)
                dataset = dataset.batch(args.batch_size)
            else:
                print(f"✗ Unsupported data format: {args.data_path}")
                return

            print("✓ Dataset loaded")
        except Exception as e:
            print(f"✗ Failed to load dataset: {e}")
            return

    # Run diagnostics
    if args.quick or dataset is None:
        # Quick check
        if dataset:
            run_quick_diagnostic(model, dataset, num_samples=1)
        else:
            print("Cannot run quick diagnostic without dataset")
    else:
        # Full diagnostic
        diagnostics = MultimodalDiagnostics(log_dir=args.output_dir, enable_tb=True)
        run_diagnostics_on_dataset(model, dataset, diagnostics, num_batches=args.num_batches)

        # Generate report
        print("\n\nGenerating diagnostic report...")
        report = diagnostics.generate_report(save_plots=True)

        print(f"\n✓ Diagnostics complete! Results saved to: {args.output_dir}")
        print(f"  - Report: {os.path.join(args.output_dir, 'diagnostic_report.json')}")
        print(f"  - Plots: {os.path.join(args.output_dir, 'plots/')}")
        print(f"  - TensorBoard: tensorboard --logdir {os.path.join(args.output_dir, 'tensorboard')}")


if __name__ == '__main__':
    # Check if running without arguments - show help
    if len(sys.argv) == 1:
        print("\n" + "="*80)
        print("DenseGNN v5 Multimodal Diagnostics")
        print("="*80)
        print("\nThis script helps diagnose why multimodal fusion hurts performance.")
        print("\nUsage examples:")
        print("\n1. Quick check (1 batch):")
        print("   python run_diagnostics_v5.py --model_path ./model.h5 \\")
        print("                                 --data_path ./test_data.pkl \\")
        print("                                 --quick")
        print("\n2. Full diagnostic (100 batches):")
        print("   python run_diagnostics_v5.py --model_path ./model.h5 \\")
        print("                                 --data_path ./test_data.pkl \\")
        print("                                 --num_batches 100 \\")
        print("                                 --output_dir ./results")
        print("\n3. View results in TensorBoard:")
        print("   tensorboard --logdir ./results/tensorboard")
        print("\n" + "="*80)
        print("\nRun with --help for full options")
        print()
        sys.exit(0)

    main()
