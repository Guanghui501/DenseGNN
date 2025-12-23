"""
Multimodal Fusion Diagnostics
=============================
Diagnostic tools to identify why multimodal fusion decreases performance.

Usage:
    from training.diagnostics.multimodal_diagnostics import MultimodalDiagnostics

    diagnostics = MultimodalDiagnostics(log_dir='./diagnostic_logs')

    # In training loop:
    diagnostics.log_embeddings(graph_emb, text_emb, epoch, batch_idx)
    diagnostics.log_fusion_gates(gate_weights, epoch, batch_idx)

    # After training:
    diagnostics.generate_report()
"""

import tensorflow as tf
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt


class MultimodalDiagnostics:
    """Diagnostic tools for multimodal fusion analysis."""

    def __init__(self, log_dir='./diagnostic_logs', enable_tb=True):
        """
        Args:
            log_dir: Directory to save diagnostic logs
            enable_tb: Enable TensorBoard logging
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Statistics storage
        self.graph_emb_stats = []
        self.text_emb_stats = []
        self.gate_weights_stats = []
        self.loss_history = {'with_text': [], 'without_text': []}

        # TensorBoard
        if enable_tb:
            tb_dir = os.path.join(log_dir, 'tensorboard')
            self.writer = tf.summary.create_file_writer(tb_dir)
        else:
            self.writer = None

        print(f"[Diagnostics] Initialized. Logs: {log_dir}")

    def log_embeddings(self, graph_emb, text_emb, epoch, batch_idx=0, step=None):
        """
        Log statistics of graph and text embeddings.

        Args:
            graph_emb: Graph embeddings tensor (batch, features)
            text_emb: Text embeddings tensor (batch, features)
            epoch: Current epoch
            batch_idx: Current batch index
            step: Global step (optional)
        """
        if step is None:
            step = epoch * 1000 + batch_idx

        # Convert to numpy if needed
        if tf.is_tensor(graph_emb):
            graph_emb = graph_emb.numpy()
        if tf.is_tensor(text_emb):
            text_emb = text_emb.numpy()

        # Compute statistics
        graph_stats = {
            'mean': float(np.mean(graph_emb)),
            'std': float(np.std(graph_emb)),
            'min': float(np.min(graph_emb)),
            'max': float(np.max(graph_emb)),
            'l2_norm': float(np.mean(np.linalg.norm(graph_emb, axis=-1))),
            'epoch': epoch,
            'batch': batch_idx,
            'step': step
        }

        text_stats = {
            'mean': float(np.mean(text_emb)),
            'std': float(np.std(text_emb)),
            'min': float(np.min(text_emb)),
            'max': float(np.max(text_emb)),
            'l2_norm': float(np.mean(np.linalg.norm(text_emb, axis=-1))),
            'epoch': epoch,
            'batch': batch_idx,
            'step': step
        }

        # Scale ratio - critical for diagnosing feature imbalance
        scale_ratio = graph_stats['std'] / (text_stats['std'] + 1e-8)
        norm_ratio = graph_stats['l2_norm'] / (text_stats['l2_norm'] + 1e-8)

        self.graph_emb_stats.append(graph_stats)
        self.text_emb_stats.append(text_stats)

        # TensorBoard logging
        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar('embeddings/graph_mean', graph_stats['mean'], step=step)
                tf.summary.scalar('embeddings/graph_std', graph_stats['std'], step=step)
                tf.summary.scalar('embeddings/graph_l2_norm', graph_stats['l2_norm'], step=step)

                tf.summary.scalar('embeddings/text_mean', text_stats['mean'], step=step)
                tf.summary.scalar('embeddings/text_std', text_stats['std'], step=step)
                tf.summary.scalar('embeddings/text_l2_norm', text_stats['l2_norm'], step=step)

                tf.summary.scalar('embeddings/std_ratio_graph_to_text', scale_ratio, step=step)
                tf.summary.scalar('embeddings/norm_ratio_graph_to_text', norm_ratio, step=step)

        # Print warning if severe imbalance detected
        if scale_ratio > 5.0 or scale_ratio < 0.2:
            print(f"[WARNING] Epoch {epoch} - Severe feature scale imbalance detected!")
            print(f"  Graph std: {graph_stats['std']:.4f}, Text std: {text_stats['std']:.4f}")
            print(f"  Ratio: {scale_ratio:.2f}x")

    def log_fusion_gates(self, gate_weights, epoch, batch_idx=0, step=None, layer_idx=None):
        """
        Log fusion gate weights to detect over-reliance on text/graph.

        Args:
            gate_weights: Gate weights tensor (batch, features) or scalar
            epoch: Current epoch
            batch_idx: Current batch index
            step: Global step
            layer_idx: Layer index for middle fusion
        """
        if step is None:
            step = epoch * 1000 + batch_idx

        if tf.is_tensor(gate_weights):
            gate_weights = gate_weights.numpy()

        # Handle both scalar and vector gates
        if np.isscalar(gate_weights):
            gate_mean = float(gate_weights)
            gate_std = 0.0
        else:
            gate_mean = float(np.mean(gate_weights))
            gate_std = float(np.std(gate_weights))

        gate_stats = {
            'mean': gate_mean,
            'std': gate_std,
            'epoch': epoch,
            'batch': batch_idx,
            'step': step,
            'layer': layer_idx
        }

        self.gate_weights_stats.append(gate_stats)

        # TensorBoard
        if self.writer is not None:
            with self.writer.as_default():
                prefix = f'fusion_gates/layer_{layer_idx}' if layer_idx is not None else 'fusion_gates'
                tf.summary.scalar(f'{prefix}/mean', gate_mean, step=step)
                if gate_std > 0:
                    tf.summary.scalar(f'{prefix}/std', gate_std, step=step)

        # Warn if gate heavily favors one modality
        if gate_mean > 0.8:
            print(f"[WARNING] Epoch {epoch} - Gate heavily favors graph (mean={gate_mean:.3f})")
        elif gate_mean < 0.2:
            print(f"[WARNING] Epoch {epoch} - Gate heavily favors text (mean={gate_mean:.3f})")

    def log_loss(self, loss, epoch, with_text=True, step=None):
        """
        Log training loss separately for with/without text.

        Args:
            loss: Loss value
            epoch: Current epoch
            with_text: Whether text was used
            step: Global step
        """
        if step is None:
            step = epoch

        if tf.is_tensor(loss):
            loss = float(loss.numpy())
        else:
            loss = float(loss)

        key = 'with_text' if with_text else 'without_text'
        self.loss_history[key].append({'epoch': epoch, 'loss': loss, 'step': step})

        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar(f'loss/{key}', loss, step=step)

    def compute_text_noise_score(self):
        """
        Compute a noise score for text embeddings.
        High variance across samples suggests noisy/inconsistent text.

        Returns:
            float: Noise score (higher = more noisy)
        """
        if len(self.text_emb_stats) < 10:
            return None

        # Variance of std across batches indicates inconsistency
        stds = [s['std'] for s in self.text_emb_stats]
        noise_score = np.std(stds) / (np.mean(stds) + 1e-8)

        return float(noise_score)

    def detect_feature_imbalance(self):
        """
        Detect if graph and text features have incompatible scales.

        Returns:
            dict: Analysis results
        """
        if len(self.graph_emb_stats) < 10 or len(self.text_emb_stats) < 10:
            return None

        graph_stds = [s['std'] for s in self.graph_emb_stats]
        text_stds = [s['std'] for s in self.text_emb_stats]

        avg_graph_std = np.mean(graph_stds)
        avg_text_std = np.mean(text_stds)
        ratio = avg_graph_std / (avg_text_std + 1e-8)

        # Check L2 norms
        graph_norms = [s['l2_norm'] for s in self.graph_emb_stats]
        text_norms = [s['l2_norm'] for s in self.text_emb_stats]
        norm_ratio = np.mean(graph_norms) / (np.mean(text_norms) + 1e-8)

        return {
            'std_ratio': float(ratio),
            'norm_ratio': float(norm_ratio),
            'graph_std': float(avg_graph_std),
            'text_std': float(avg_text_std),
            'graph_norm': float(np.mean(graph_norms)),
            'text_norm': float(np.mean(text_norms)),
            'imbalanced': ratio > 3.0 or ratio < 0.33  # Flag if >3x difference
        }

    def detect_overfitting_to_text(self):
        """
        Detect if model overly relies on text modality.

        Returns:
            dict: Analysis results
        """
        if len(self.gate_weights_stats) < 10:
            return None

        gate_means = [s['mean'] for s in self.gate_weights_stats]
        avg_gate = np.mean(gate_means)
        gate_trend = np.polyfit(range(len(gate_means)), gate_means, deg=1)[0]

        return {
            'avg_gate_weight': float(avg_gate),
            'gate_trend': float(gate_trend),  # Positive = favoring graph over time
            'text_dominant': avg_gate < 0.3,  # Gate < 0.3 means text dominates
            'graph_dominant': avg_gate > 0.7,  # Gate > 0.7 means graph dominates
        }

    def generate_report(self, save_plots=True):
        """
        Generate comprehensive diagnostic report.

        Args:
            save_plots: Whether to save visualization plots

        Returns:
            dict: Diagnostic report
        """
        print("\n" + "="*80)
        print("MULTIMODAL FUSION DIAGNOSTIC REPORT")
        print("="*80)

        report = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(self.graph_emb_stats)
        }

        # 1. Feature imbalance analysis
        print("\n[1] FEATURE SCALE IMBALANCE CHECK")
        print("-" * 80)
        imbalance = self.detect_feature_imbalance()
        if imbalance:
            report['feature_imbalance'] = imbalance
            print(f"Graph embedding std:  {imbalance['graph_std']:.4f}")
            print(f"Text embedding std:   {imbalance['text_std']:.4f}")
            print(f"Std ratio (G/T):      {imbalance['std_ratio']:.2f}x")
            print(f"Norm ratio (G/T):     {imbalance['norm_ratio']:.2f}x")

            if imbalance['imbalanced']:
                print("\n🔴 PROBLEM DETECTED: Severe feature scale imbalance!")
                print("   → Graph and text embeddings have incompatible scales")
                print("   → Fusion will be dominated by the larger-scale modality")
                print("\n   Solutions:")
                print("   - Add LayerNormalization before fusion")
                print("   - Use separate projection heads with normalization")
                print("   - Scale text embeddings: text_emb = text_emb * scale_factor")
            else:
                print("\n✅ Feature scales are balanced")
        else:
            print("Insufficient data for analysis")

        # 2. Text noise analysis
        print("\n[2] TEXT EMBEDDING QUALITY CHECK")
        print("-" * 80)
        noise_score = self.compute_text_noise_score()
        if noise_score is not None:
            report['text_noise_score'] = noise_score
            print(f"Text embedding noise score: {noise_score:.4f}")

            if noise_score > 0.5:
                print("\n🔴 PROBLEM DETECTED: High text embedding variance!")
                print("   → Text embeddings are inconsistent across samples")
                print("   → May indicate poor text quality or encoding issues")
                print("\n   Solutions:")
                print("   - Review text descriptions for quality")
                print("   - Fine-tune BERT encoder (if frozen)")
                print("   - Add text embedding dropout for regularization")
            else:
                print("✅ Text embeddings appear consistent")
        else:
            print("Insufficient data for analysis")

        # 3. Overfitting analysis
        print("\n[3] TEXT OVER-RELIANCE CHECK")
        print("-" * 80)
        overfitting = self.detect_overfitting_to_text()
        if overfitting:
            report['overfitting_analysis'] = overfitting
            print(f"Average gate weight (0=text, 1=graph): {overfitting['avg_gate_weight']:.3f}")
            print(f"Gate weight trend: {overfitting['gate_trend']:.4f}")

            if overfitting['text_dominant']:
                print("\n🔴 PROBLEM DETECTED: Model over-relies on text!")
                print("   → Gate weights heavily favor text modality")
                print("   → Model may ignore graph structure")
                print("\n   Solutions:")
                print("   - Increase graph embedding dimension")
                print("   - Add modality-specific dropout (higher for text)")
                print("   - Use contrastive learning to align modalities")
            elif overfitting['graph_dominant']:
                print("\n⚠️  Model ignores text (gate favors graph)")
                print("   → Text features may be uninformative or noisy")
            else:
                print("✅ Balanced fusion between graph and text")
        else:
            print("No gate weights logged - late fusion only?")

        # 4. Loss comparison
        print("\n[4] LOSS CURVE COMPARISON")
        print("-" * 80)
        if self.loss_history['with_text'] and self.loss_history['without_text']:
            loss_with = [l['loss'] for l in self.loss_history['with_text']]
            loss_without = [l['loss'] for l in self.loss_history['without_text']]

            avg_with = np.mean(loss_with[-10:]) if len(loss_with) >= 10 else np.mean(loss_with)
            avg_without = np.mean(loss_without[-10:]) if len(loss_without) >= 10 else np.mean(loss_without)

            print(f"Avg loss with text:    {avg_with:.4f}")
            print(f"Avg loss without text: {avg_without:.4f}")
            print(f"Difference:            {avg_with - avg_without:+.4f}")

            if avg_with > avg_without:
                print("\n🔴 PROBLEM DETECTED: Adding text increases loss!")
                print("   → Text is hurting performance, not helping")
        else:
            print("Run ablation study: train with/without text to compare")

        # Save report
        report_path = os.path.join(self.log_dir, 'diagnostic_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📊 Full report saved to: {report_path}")

        # Generate plots
        if save_plots:
            self._generate_plots()

        print("\n" + "="*80)
        return report

    def _generate_plots(self):
        """Generate visualization plots."""
        plot_dir = os.path.join(self.log_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        # Plot 1: Embedding statistics over time
        if self.graph_emb_stats and self.text_emb_stats:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            steps = [s['step'] for s in self.graph_emb_stats]
            graph_stds = [s['std'] for s in self.graph_emb_stats]
            text_stds = [s['std'] for s in self.text_emb_stats]
            graph_norms = [s['l2_norm'] for s in self.graph_emb_stats]
            text_norms = [s['l2_norm'] for s in self.text_emb_stats]

            axes[0, 0].plot(steps, graph_stds, label='Graph', alpha=0.7)
            axes[0, 0].plot(steps, text_stds, label='Text', alpha=0.7)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Std Dev')
            axes[0, 0].set_title('Embedding Standard Deviation')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(steps, graph_norms, label='Graph', alpha=0.7)
            axes[0, 1].plot(steps, text_norms, label='Text', alpha=0.7)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('L2 Norm')
            axes[0, 1].set_title('Embedding L2 Norm')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Ratios
            std_ratios = [g/t if t > 0 else 0 for g, t in zip(graph_stds, text_stds)]
            norm_ratios = [g/t if t > 0 else 0 for g, t in zip(graph_norms, text_norms)]

            axes[1, 0].plot(steps, std_ratios, alpha=0.7, color='purple')
            axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Balanced')
            axes[1, 0].axhline(y=3.0, color='orange', linestyle='--', alpha=0.5, label='Threshold')
            axes[1, 0].axhline(y=0.33, color='orange', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Ratio (Graph/Text)')
            axes[1, 0].set_title('Std Dev Ratio')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(steps, norm_ratios, alpha=0.7, color='green')
            axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Balanced')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Ratio (Graph/Text)')
            axes[1, 1].set_title('L2 Norm Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'embedding_statistics.png'), dpi=150)
            plt.close()
            print(f"📈 Plot saved: {plot_dir}/embedding_statistics.png")

        # Plot 2: Gate weights over time
        if self.gate_weights_stats:
            fig, ax = plt.subplots(figsize=(10, 5))

            steps = [s['step'] for s in self.gate_weights_stats]
            gates = [s['mean'] for s in self.gate_weights_stats]

            ax.plot(steps, gates, alpha=0.7, linewidth=2)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Balanced')
            ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Text-dominant threshold')
            ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Graph-dominant threshold')
            ax.fill_between(steps, 0.3, 0.7, alpha=0.1, color='green', label='Healthy range')
            ax.set_xlabel('Step')
            ax.set_ylabel('Gate Weight (0=text, 1=graph)')
            ax.set_title('Fusion Gate Weights Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'gate_weights.png'), dpi=150)
            plt.close()
            print(f"📈 Plot saved: {plot_dir}/gate_weights.png")


# Convenience function for quick diagnostics
def quick_diagnostic_check(graph_emb, text_emb, gate_weight=None):
    """
    Quick one-off diagnostic check without full logging infrastructure.

    Args:
        graph_emb: Graph embeddings
        text_emb: Text embeddings
        gate_weight: Optional gate weight

    Returns:
        dict: Diagnostic results
    """
    if tf.is_tensor(graph_emb):
        graph_emb = graph_emb.numpy()
    if tf.is_tensor(text_emb):
        text_emb = text_emb.numpy()

    graph_std = np.std(graph_emb)
    text_std = np.std(text_emb)
    ratio = graph_std / (text_std + 1e-8)

    graph_norm = np.mean(np.linalg.norm(graph_emb, axis=-1))
    text_norm = np.mean(np.linalg.norm(text_emb, axis=-1))
    norm_ratio = graph_norm / (text_norm + 1e-8)

    results = {
        'graph_std': float(graph_std),
        'text_std': float(text_std),
        'std_ratio': float(ratio),
        'graph_norm': float(graph_norm),
        'text_norm': float(text_norm),
        'norm_ratio': float(norm_ratio),
        'imbalanced': ratio > 3.0 or ratio < 0.33
    }

    if gate_weight is not None:
        if tf.is_tensor(gate_weight):
            gate_weight = gate_weight.numpy()
        results['gate_weight'] = float(np.mean(gate_weight))

    print("\n" + "="*60)
    print("QUICK DIAGNOSTIC CHECK")
    print("="*60)
    print(f"Graph embedding std:  {results['graph_std']:.4f}")
    print(f"Text embedding std:   {results['text_std']:.4f}")
    print(f"Std ratio (G/T):      {results['std_ratio']:.2f}x")
    print(f"Norm ratio (G/T):     {results['norm_ratio']:.2f}x")

    if results['imbalanced']:
        print("\n🔴 WARNING: Feature scale imbalance detected!")
    else:
        print("\n✅ Feature scales are balanced")

    if 'gate_weight' in results:
        print(f"\nGate weight (0=text, 1=graph): {results['gate_weight']:.3f}")
        if results['gate_weight'] < 0.3:
            print("🔴 Model heavily relies on text!")
        elif results['gate_weight'] > 0.7:
            print("⚠️  Model ignores text!")

    print("="*60 + "\n")

    return results
