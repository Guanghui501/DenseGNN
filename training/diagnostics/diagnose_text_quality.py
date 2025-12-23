"""
Advanced Text Quality Diagnostics

This script performs deep analysis of text embeddings to identify:
1. Text-label correlation (potential label leakage)
2. Text semantic quality
3. Text-graph alignment
4. Information content in text vs graph
"""

import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os


class TextQualityDiagnostics:
    """Advanced diagnostics for text quality issues."""

    def __init__(self, log_dir='./text_quality_diagnostics'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.graph_predictions = []
        self.text_predictions = []
        self.labels = []
        self.gate_weights = []

        print(f"[Text Quality Diagnostics] Initialized. Logs: {log_dir}")

    def analyze_text_label_correlation(self, text_embeddings, labels):
        """
        Check if text embeddings directly correlate with labels.
        High correlation suggests label leakage or overfitting to text.

        Args:
            text_embeddings: Text embeddings (N, text_dim)
            labels: Target values (N,)

        Returns:
            dict: Correlation analysis
        """
        if tf.is_tensor(text_embeddings):
            text_embeddings = text_embeddings.numpy()
        if tf.is_tensor(labels):
            labels = labels.numpy()

        # Flatten if needed
        if len(labels.shape) > 1:
            labels = labels.flatten()

        # Compute correlation between each text dimension and labels
        correlations = []
        for i in range(text_embeddings.shape[1]):
            try:
                corr, _ = pearsonr(text_embeddings[:, i], labels)
                correlations.append(abs(corr))
            except:
                correlations.append(0.0)

        correlations = np.array(correlations)

        # Statistics
        max_corr = np.max(correlations)
        mean_corr = np.mean(correlations)
        num_high_corr = np.sum(correlations > 0.3)  # Dimensions with >0.3 correlation

        # Fit simple linear model: text -> label
        model = Ridge(alpha=1.0)
        model.fit(text_embeddings, labels)
        text_only_predictions = model.predict(text_embeddings)
        text_only_mae = mean_absolute_error(labels, text_only_predictions)

        results = {
            'max_correlation': float(max_corr),
            'mean_correlation': float(mean_corr),
            'num_high_corr_dims': int(num_high_corr),
            'text_only_mae': float(text_only_mae),
            'correlation_distribution': correlations.tolist()
        }

        return results

    def analyze_graph_text_complementarity(self, graph_embeddings, text_embeddings, labels):
        """
        Analyze if text provides complementary information to graph.

        Tests:
        1. Graph-only prediction performance
        2. Text-only prediction performance
        3. Combined prediction performance
        4. Information overlap

        Args:
            graph_embeddings: Graph embeddings (N, graph_dim)
            text_embeddings: Text embeddings (N, text_dim)
            labels: Target values (N,)

        Returns:
            dict: Complementarity analysis
        """
        if tf.is_tensor(graph_embeddings):
            graph_embeddings = graph_embeddings.numpy()
        if tf.is_tensor(text_embeddings):
            text_embeddings = text_embeddings.numpy()
        if tf.is_tensor(labels):
            labels = labels.numpy()

        if len(labels.shape) > 1:
            labels = labels.flatten()

        # Split data for validation
        n = len(labels)
        train_idx = int(0.8 * n)

        g_train, g_val = graph_embeddings[:train_idx], graph_embeddings[train_idx:]
        t_train, t_val = text_embeddings[:train_idx], text_embeddings[train_idx:]
        y_train, y_val = labels[:train_idx], labels[train_idx:]

        # 1. Graph-only model
        model_g = Ridge(alpha=1.0)
        model_g.fit(g_train, y_train)
        pred_g = model_g.predict(g_val)
        mae_graph_only = mean_absolute_error(y_val, pred_g)

        # 2. Text-only model
        model_t = Ridge(alpha=1.0)
        model_t.fit(t_train, y_train)
        pred_t = model_t.predict(t_val)
        mae_text_only = mean_absolute_error(y_val, pred_t)

        # 3. Combined model
        combined_train = np.concatenate([g_train, t_train], axis=-1)
        combined_val = np.concatenate([g_val, t_val], axis=-1)
        model_c = Ridge(alpha=1.0)
        model_c.fit(combined_train, y_train)
        pred_c = model_c.predict(combined_val)
        mae_combined = mean_absolute_error(y_val, pred_c)

        # 4. Compute information overlap (correlation between graph and text predictions)
        pred_g_full = model_g.predict(graph_embeddings)
        pred_t_full = model_t.predict(text_embeddings)

        try:
            overlap_corr, _ = pearsonr(pred_g_full, pred_t_full)
        except:
            overlap_corr = 0.0

        # Analysis
        results = {
            'mae_graph_only': float(mae_graph_only),
            'mae_text_only': float(mae_text_only),
            'mae_combined': float(mae_combined),
            'improvement_over_graph': float(mae_graph_only - mae_combined),
            'text_helps': mae_combined < mae_graph_only,
            'prediction_overlap_correlation': float(overlap_corr),
            'text_information_quality': 'good' if mae_text_only < mae_graph_only * 1.5 else 'poor'
        }

        return results

    def analyze_text_embedding_variance(self, text_embeddings, batch_labels=None):
        """
        Analyze variance structure of text embeddings.

        High variance across samples = good (diverse information)
        Low variance across samples = bad (uninformative)

        Args:
            text_embeddings: Text embeddings (N, text_dim)
            batch_labels: Optional batch IDs to check within-batch variance

        Returns:
            dict: Variance analysis
        """
        if tf.is_tensor(text_embeddings):
            text_embeddings = text_embeddings.numpy()

        # Overall variance
        feature_vars = np.var(text_embeddings, axis=0)
        mean_var = np.mean(feature_vars)
        std_var = np.std(feature_vars)

        # Check if variance is concentrated in few dimensions
        sorted_vars = np.sort(feature_vars)[::-1]
        top_10_pct_var = np.sum(sorted_vars[:len(sorted_vars)//10])
        total_var = np.sum(feature_vars)
        var_concentration = top_10_pct_var / (total_var + 1e-8)

        # Sample-to-sample variance (diversity)
        sample_norms = np.linalg.norm(text_embeddings, axis=1)
        norm_var = np.var(sample_norms)

        # Pairwise cosine similarity distribution
        n_samples = min(1000, len(text_embeddings))  # Sample for efficiency
        sample_idx = np.random.choice(len(text_embeddings), n_samples, replace=False)
        sampled = text_embeddings[sample_idx]

        # Normalize
        sampled_norm = sampled / (np.linalg.norm(sampled, axis=1, keepdims=True) + 1e-8)

        # Compute similarities
        similarities = np.dot(sampled_norm, sampled_norm.T)
        # Remove diagonal
        similarities = similarities[~np.eye(n_samples, dtype=bool)]

        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)

        results = {
            'mean_feature_variance': float(mean_var),
            'std_feature_variance': float(std_var),
            'variance_concentration_top_10pct': float(var_concentration),
            'sample_norm_variance': float(norm_var),
            'mean_cosine_similarity': float(mean_similarity),
            'std_cosine_similarity': float(std_similarity),
            'embedding_diversity': 'good' if mean_similarity < 0.7 else 'poor'
        }

        return results

    def check_text_noise_via_nearest_neighbors(self, text_embeddings, labels, k=5):
        """
        Check if similar text embeddings have similar labels.
        If not, text is noisy/unreliable.

        Args:
            text_embeddings: Text embeddings (N, text_dim)
            labels: Target values (N,)
            k: Number of nearest neighbors

        Returns:
            dict: Noise analysis
        """
        if tf.is_tensor(text_embeddings):
            text_embeddings = text_embeddings.numpy()
        if tf.is_tensor(labels):
            labels = labels.numpy()

        if len(labels.shape) > 1:
            labels = labels.flatten()

        # Normalize embeddings
        text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)

        # For each sample, find k nearest neighbors
        n_samples = min(500, len(text_embeddings))  # Sample for efficiency
        sample_idx = np.random.choice(len(text_embeddings), n_samples, replace=False)

        label_consistencies = []

        for i in sample_idx:
            # Compute similarities to all other samples
            sims = np.dot(text_norm, text_norm[i])

            # Get k nearest neighbors (excluding self)
            nn_idx = np.argsort(sims)[::-1][1:k+1]

            # Check label consistency
            label_i = labels[i]
            labels_nn = labels[nn_idx]

            # Compute MAE between this label and neighbors
            consistency = np.mean(np.abs(labels_nn - label_i))
            label_consistencies.append(consistency)

        mean_consistency = np.mean(label_consistencies)
        std_consistency = np.std(label_consistencies)

        # Compare to random baseline
        random_pairs = np.random.choice(len(labels), (len(label_consistencies), k))
        random_consistencies = []
        for i, pair in enumerate(random_pairs):
            random_consistencies.append(np.mean(np.abs(labels[pair] - labels[sample_idx[i]])))

        mean_random = np.mean(random_consistencies)

        # Good text: similar embeddings -> similar labels (low consistency MAE)
        # Bad text: similar embeddings -> different labels (high consistency MAE)

        results = {
            'mean_neighbor_label_mae': float(mean_consistency),
            'std_neighbor_label_mae': float(std_consistency),
            'random_baseline_mae': float(mean_random),
            'improvement_over_random': float(mean_random - mean_consistency),
            'text_is_informative': mean_consistency < mean_random * 0.8
        }

        return results

    def generate_text_quality_report(self, graph_embeddings, text_embeddings, labels):
        """
        Generate comprehensive text quality report.

        Args:
            graph_embeddings: Graph embeddings
            text_embeddings: Text embeddings
            labels: Target labels

        Returns:
            dict: Full diagnostic report
        """
        print("\n" + "="*80)
        print("TEXT QUALITY DIAGNOSTIC REPORT")
        print("="*80)

        report = {}

        # 1. Text-Label Correlation Check
        print("\n[1] TEXT-LABEL CORRELATION CHECK")
        print("-" * 80)
        correlation_results = self.analyze_text_label_correlation(text_embeddings, labels)
        report['text_label_correlation'] = correlation_results

        print(f"Max correlation (single dimension): {correlation_results['max_correlation']:.3f}")
        print(f"Mean correlation:                   {correlation_results['mean_correlation']:.3f}")
        print(f"High-correlation dimensions:        {correlation_results['num_high_corr_dims']}")
        print(f"Text-only linear model MAE:         {correlation_results['text_only_mae']:.2f}")

        if correlation_results['max_correlation'] > 0.5:
            print("\n🔴 PROBLEM: Strong text-label correlation detected!")
            print("   → Text may contain label leakage")
            print("   → Model shortcuts through text, ignores graph")
            print("\n   Check if text descriptions mention target values!")
        elif correlation_results['num_high_corr_dims'] > 10:
            print("\n⚠️  Many dimensions correlate with labels")
            print("   → Text may encode target information")
        else:
            print("\n✅ Text-label correlation is healthy")

        # 2. Graph-Text Complementarity
        print("\n[2] GRAPH-TEXT COMPLEMENTARITY CHECK")
        print("-" * 80)
        comp_results = self.analyze_graph_text_complementarity(
            graph_embeddings, text_embeddings, labels
        )
        report['complementarity'] = comp_results

        print(f"MAE (Graph only):  {comp_results['mae_graph_only']:.2f}")
        print(f"MAE (Text only):   {comp_results['mae_text_only']:.2f}")
        print(f"MAE (Combined):    {comp_results['mae_combined']:.2f}")
        print(f"Improvement:       {comp_results['improvement_over_graph']:+.2f}")
        print(f"Prediction overlap correlation: {comp_results['prediction_overlap_correlation']:.3f}")

        if not comp_results['text_helps']:
            print("\n🔴 CRITICAL PROBLEM: Text does NOT help!")
            print("   → Text is adding noise, not information")
            print("   → Combined model worse than graph-only")
            print("\n   Possible causes:")
            print("   - Text quality is poor")
            print("   - Text is not relevant to prediction task")
            print("   - Text-graph semantic mismatch")
        elif comp_results['improvement_over_graph'] < 0.5:
            print("\n⚠️  Text provides minimal improvement")
            print("   → Text adds little information beyond graph")
        else:
            print("\n✅ Text provides complementary information")

        if comp_results['mae_text_only'] > comp_results['mae_graph_only'] * 2:
            print("\n🔴 Text-only model is very poor!")
            print(f"   → Text MAE ({comp_results['mae_text_only']:.2f}) >> Graph MAE ({comp_results['mae_graph_only']:.2f})")
            print("   → Text does not contain useful information for this task")

        # 3. Text Embedding Variance
        print("\n[3] TEXT EMBEDDING VARIANCE CHECK")
        print("-" * 80)
        var_results = self.analyze_text_embedding_variance(text_embeddings)
        report['variance'] = var_results

        print(f"Mean feature variance:          {var_results['mean_feature_variance']:.4f}")
        print(f"Variance concentration (top 10%): {var_results['variance_concentration_top_10pct']:.3f}")
        print(f"Mean cosine similarity:         {var_results['mean_cosine_similarity']:.3f}")
        print(f"Embedding diversity:            {var_results['embedding_diversity']}")

        if var_results['embedding_diversity'] == 'poor':
            print("\n🔴 PROBLEM: Text embeddings are too similar!")
            print("   → All samples have similar text embeddings")
            print("   → Text is not discriminative")
            print("\n   Possible causes:")
            print("   - Text descriptions are generic/template-based")
            print("   - BERT encoder is frozen and not adapted")
            print("   - Text preprocessing removes important information")
        else:
            print("\n✅ Text embeddings are diverse")

        # 4. Text Noise via Nearest Neighbors
        print("\n[4] TEXT NOISE CHECK (Nearest Neighbors)")
        print("-" * 80)
        noise_results = self.check_text_noise_via_nearest_neighbors(
            text_embeddings, labels, k=5
        )
        report['noise_analysis'] = noise_results

        print(f"Neighbor label MAE:       {noise_results['mean_neighbor_label_mae']:.2f}")
        print(f"Random baseline MAE:      {noise_results['random_baseline_mae']:.2f}")
        print(f"Improvement over random:  {noise_results['improvement_over_random']:.2f}")
        print(f"Text is informative:      {noise_results['text_is_informative']}")

        if not noise_results['text_is_informative']:
            print("\n🔴 PROBLEM: Text is noisy/unreliable!")
            print("   → Similar text embeddings have different labels")
            print("   → Text does not reliably encode semantic information")
            print("\n   Possible causes:")
            print("   - Auto-generated text has inconsistencies")
            print("   - Text descriptions are inaccurate")
            print("   - BERT fails to capture relevant semantics")
        else:
            print("\n✅ Text embeddings are informative")

        # Final Summary
        print("\n" + "="*80)
        print("SUMMARY & RECOMMENDATIONS")
        print("="*80)

        problems = []
        if correlation_results['max_correlation'] > 0.5:
            problems.append("label_leakage")
        if not comp_results['text_helps']:
            problems.append("text_hurts_performance")
        if comp_results['mae_text_only'] > comp_results['mae_graph_only'] * 2:
            problems.append("text_not_predictive")
        if var_results['embedding_diversity'] == 'poor':
            problems.append("text_not_diverse")
        if not noise_results['text_is_informative']:
            problems.append("text_noisy")

        report['problems_detected'] = problems
        report['num_problems'] = len(problems)

        if len(problems) == 0:
            print("\n✅ No major text quality issues detected")
            print("   → Problem likely in fusion mechanism or training strategy")
        else:
            print(f"\n🔴 Detected {len(problems)} text quality issue(s):\n")

            for problem in problems:
                if problem == "label_leakage":
                    print("  1. LABEL LEAKAGE")
                    print("     → Text contains information about target labels")
                    print("     → Solution: Review text generation, remove target mentions")

                elif problem == "text_hurts_performance":
                    print("  2. TEXT HURTS PERFORMANCE")
                    print("     → Multimodal worse than graph-only in simple linear model")
                    print("     → Solution: Consider removing text entirely, or use only late fusion")

                elif problem == "text_not_predictive":
                    print("  3. TEXT NOT PREDICTIVE")
                    print("     → Text-only model performs very poorly")
                    print("     → Solution: Improve text quality or use different text source")

                elif problem == "text_not_diverse":
                    print("  4. TEXT NOT DIVERSE")
                    print("     → All text embeddings are similar")
                    print("     → Solution: Fine-tune BERT, improve text generation diversity")

                elif problem == "text_noisy":
                    print("  5. TEXT IS NOISY")
                    print("     → Similar text embeddings -> different labels")
                    print("     → Solution: Clean text data, add dropout, use contrastive learning")

        # Save report
        import json
        report_path = os.path.join(self.log_dir, 'text_quality_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Full report saved to: {report_path}")
        print("="*80 + "\n")

        return report


def quick_text_quality_check(graph_emb, text_emb, labels):
    """
    Quick text quality check without full infrastructure.

    Args:
        graph_emb: Graph embeddings
        text_emb: Text embeddings
        labels: Target labels

    Returns:
        dict: Quick diagnostic results
    """
    diagnostics = TextQualityDiagnostics(log_dir='/tmp/text_quality_check')

    # Run only essential checks
    print("\n" + "="*80)
    print("QUICK TEXT QUALITY CHECK")
    print("="*80)

    # Check 1: Does text help at all?
    comp = diagnostics.analyze_graph_text_complementarity(graph_emb, text_emb, labels)

    print(f"\nGraph-only MAE:  {comp['mae_graph_only']:.2f}")
    print(f"Text-only MAE:   {comp['mae_text_only']:.2f}")
    print(f"Combined MAE:    {comp['mae_combined']:.2f}")

    if comp['text_helps']:
        print(f"\n✅ Text helps: Improvement of {comp['improvement_over_graph']:.2f} MAE")
    else:
        print(f"\n🔴 PROBLEM: Text hurts performance by {abs(comp['improvement_over_graph']):.2f} MAE")
        print("   → Your multimodal model will be worse than graph-only!")

    # Check 2: Is text informative?
    noise = diagnostics.check_text_noise_via_nearest_neighbors(text_emb, labels)

    if noise['text_is_informative']:
        print(f"\n✅ Text is informative (neighbor MAE: {noise['mean_neighbor_label_mae']:.2f})")
    else:
        print(f"\n🔴 PROBLEM: Text is noisy (neighbor MAE: {noise['mean_neighbor_label_mae']:.2f})")
        print("   → Similar text -> different labels")

    print("="*80 + "\n")

    return {'complementarity': comp, 'noise': noise}


if __name__ == '__main__':
    print("Text Quality Diagnostics Tool")
    print("\nUsage:")
    print("  from training.diagnostics.diagnose_text_quality import TextQualityDiagnostics")
    print("  diagnostics = TextQualityDiagnostics()")
    print("  report = diagnostics.generate_text_quality_report(graph_emb, text_emb, labels)")
