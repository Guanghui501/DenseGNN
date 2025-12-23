"""
Quick fix applicator for multimodal fusion issues

This script helps you apply recommended fixes based on diagnostic results.

Usage:
    python apply_fixes.py --diagnostic_report ./diagnostic_results/diagnostic_report.json \\
                          --model_file ../../kgcnn/literature/DenseGNN/_make_dense_multimodal_v5.py \\
                          --output_file ../../kgcnn/literature/DenseGNN/_make_dense_multimodal_v5_fixed.py
"""

import argparse
import json
import os
import re


class FixApplicator:
    """Applies fixes to model file based on diagnostic results."""

    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        with open(model_file_path, 'r') as f:
            self.content = f.read()
            self.lines = self.content.split('\n')

    def apply_fix_layer_normalization(self):
        """Add LayerNormalization to graph and text embeddings."""

        print("\n[Fix #1] Adding LayerNormalization...")

        # Add import if not present
        if 'from tensorflow.keras.layers import LayerNormalization' not in self.content:
            # Find imports section
            import_line = None
            for i, line in enumerate(self.lines):
                if 'from tensorflow.keras' in line or 'import tensorflow as tf' in line:
                    import_line = i
                    break

            if import_line:
                self.lines.insert(import_line + 1, 'from tensorflow.keras.layers import LayerNormalization')
                print("  ✓ Added LayerNormalization import")
            else:
                print("  ⚠ Could not find import section")
                return False

        # Find graph_emb and text_emb projections and add normalization

        # Pattern 1: Find "graph_emb = graph_projection(...)"
        graph_emb_pattern = r'^(\s*)graph_emb = graph_projection\('
        text_emb_pattern = r'^(\s*)text_emb = text_projection\('

        graph_fixed = False
        text_fixed = False

        for i, line in enumerate(self.lines):
            # Check for graph embedding
            match = re.match(graph_emb_pattern, line)
            if match and 'LayerNormalization' not in self.lines[i+1]:
                indent = match.group(1)
                # Insert normalization after this line
                self.lines.insert(i + 1, f"{indent}graph_emb = LayerNormalization(name='graph_norm')(graph_emb)  # [DIAGNOSTIC FIX]")
                graph_fixed = True
                print("  ✓ Added graph embedding normalization")

            # Check for text embedding
            match = re.match(text_emb_pattern, line)
            if match and 'LayerNormalization' not in self.lines[i+1]:
                indent = match.group(1)
                self.lines.insert(i + 1, f"{indent}text_emb = LayerNormalization(name='text_norm')(text_emb)  # [DIAGNOSTIC FIX]")
                text_fixed = True
                print("  ✓ Added text embedding normalization")

        if graph_fixed and text_fixed:
            print("  ✅ Layer normalization added successfully")
            return True
        else:
            print("  ⚠ Could not find embedding locations")
            return False

    def apply_fix_modality_dropout(self, text_dropout=0.3, graph_dropout=0.1):
        """Add modality-specific dropout."""

        print(f"\n[Fix #2] Adding modality-specific dropout (text={text_dropout}, graph={graph_dropout})...")

        # Add Dropout import if needed
        if 'from tensorflow.keras.layers import' in self.content and 'Dropout' not in self.content:
            for i, line in enumerate(self.lines):
                if 'from tensorflow.keras.layers import LayerNormalization' in line:
                    self.lines[i] = line.replace('LayerNormalization', 'LayerNormalization, Dropout')
                    print("  ✓ Added Dropout import")
                    break

        # Add dropout after normalization (or after projection if no normalization)
        graph_dropout_pattern = r'^(\s*)graph_emb = LayerNormalization.*graph_emb\)'
        text_dropout_pattern = r'^(\s*)text_emb = LayerNormalization.*text_emb\)'

        graph_fixed = False
        text_fixed = False

        for i, line in enumerate(self.lines):
            # Add graph dropout
            if re.match(graph_dropout_pattern, line) and 'Dropout' not in self.lines[i+1]:
                indent = re.match(r'^(\s*)', line).group(1)
                self.lines.insert(i + 1, f"{indent}graph_emb = Dropout({graph_dropout})(graph_emb)  # [DIAGNOSTIC FIX]")
                graph_fixed = True
                print(f"  ✓ Added graph dropout ({graph_dropout})")

            # Add text dropout
            if re.match(text_dropout_pattern, line) and 'Dropout' not in self.lines[i+1]:
                indent = re.match(r'^(\s*)', line).group(1)
                self.lines.insert(i + 1, f"{indent}text_emb = Dropout({text_dropout})(text_emb)  # [DIAGNOSTIC FIX]")
                text_fixed = True
                print(f"  ✓ Added text dropout ({text_dropout})")

        if graph_fixed and text_fixed:
            print("  ✅ Dropout added successfully")
            return True
        else:
            print("  ⚠ Could not find normalization locations")
            print("  💡 Tip: Apply Fix #1 (layer normalization) first")
            return False

    def apply_fix_expose_embeddings(self):
        """Modify model to output embeddings for diagnostics."""

        print("\n[Fix #3] Exposing embeddings for diagnostics...")

        # Find the return statement
        return_pattern = r'^(\s*)return ks\.Model\(inputs=input_list, outputs=out, name=name\)'

        for i, line in enumerate(self.lines):
            match = re.match(return_pattern, line)
            if match:
                indent = match.group(1)

                # Replace with multi-output return
                new_return = f'''{indent}return ks.Model(
{indent}    inputs=input_list,
{indent}    outputs={{'prediction': out, 'graph_emb': graph_emb, 'text_emb': text_emb}},  # [DIAGNOSTIC FIX]
{indent}    name=name
{indent})'''

                self.lines[i] = new_return
                print("  ✓ Modified return statement to expose embeddings")
                print("  ✅ Embeddings now accessible via: outputs['graph_emb'], outputs['text_emb']")
                return True

        print("  ⚠ Could not find return statement")
        return False

    def apply_fix_increase_graph_dim(self, new_graph_dim=256):
        """Increase graph projection dimension."""

        print(f"\n[Fix #4] Increasing graph projection dimension to {new_graph_dim}...")

        # This fix should be applied to the config file, not the model file
        print("  ℹ️  This fix should be applied to your config file:")
        print(f"     Change: graph_projection_dim = {new_graph_dim}  # was 128")
        print("  ⚠ Cannot auto-apply - please modify config manually")
        return False

    def save(self, output_path):
        """Save modified content."""
        self.content = '\n'.join(self.lines)

        with open(output_path, 'w') as f:
            f.write(self.content)

        print(f"\n✅ Modified model saved to: {output_path}")

    def preview_changes(self):
        """Show what will be changed."""
        print("\n" + "="*80)
        print("PREVIEW OF CHANGES")
        print("="*80)

        for i, line in enumerate(self.lines):
            if '[DIAGNOSTIC FIX]' in line:
                print(f"\nLine {i+1}:")
                print(f"  {line}")


def load_diagnostic_report(report_path):
    """Load diagnostic report and determine which fixes to apply."""

    if not os.path.exists(report_path):
        return None

    with open(report_path, 'r') as f:
        report = json.load(f)

    recommendations = []

    # Check feature imbalance
    if 'feature_imbalance' in report:
        imbalance = report['feature_imbalance']
        if imbalance.get('imbalanced', False):
            recommendations.append({
                'fix': 'layer_normalization',
                'priority': 'HIGH',
                'reason': f"Feature scale imbalance detected (ratio: {imbalance['std_ratio']:.2f}x)"
            })

    # Check text over-reliance
    if 'overfitting_analysis' in report:
        overfitting = report['overfitting_analysis']
        if overfitting.get('text_dominant', False):
            recommendations.append({
                'fix': 'modality_dropout',
                'priority': 'MEDIUM',
                'reason': f"Model over-relies on text (gate: {overfitting['avg_gate_weight']:.3f})"
            })

            recommendations.append({
                'fix': 'increase_graph_dim',
                'priority': 'MEDIUM',
                'reason': "Increase graph capacity to compete with text"
            })

    # Always recommend exposing embeddings for future diagnostics
    recommendations.append({
        'fix': 'expose_embeddings',
        'priority': 'LOW',
        'reason': "Enable future diagnostic monitoring"
    })

    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Apply fixes to multimodal model')
    parser.add_argument('--diagnostic_report', type=str, help='Path to diagnostic_report.json')
    parser.add_argument('--model_file', type=str, required=True, help='Path to model file')
    parser.add_argument('--output_file', type=str, help='Output path (default: model_file_fixed.py)')
    parser.add_argument('--preview', action='store_true', help='Preview changes without saving')
    parser.add_argument('--apply_all', action='store_true', help='Apply all fixes')

    # Individual fix flags
    parser.add_argument('--fix_normalization', action='store_true', help='Apply layer normalization')
    parser.add_argument('--fix_dropout', action='store_true', help='Apply modality dropout')
    parser.add_argument('--fix_embeddings', action='store_true', help='Expose embeddings')

    args = parser.parse_args()

    # Determine output path
    if not args.output_file:
        base, ext = os.path.splitext(args.model_file)
        args.output_file = f"{base}_fixed{ext}"

    print("\n" + "="*80)
    print("MULTIMODAL FUSION FIX APPLICATOR")
    print("="*80)

    # Load model file
    print(f"\nLoading model file: {args.model_file}")
    applicator = FixApplicator(args.model_file)

    # Determine which fixes to apply
    fixes_to_apply = []

    if args.diagnostic_report:
        print(f"Loading diagnostic report: {args.diagnostic_report}")
        recommendations = load_diagnostic_report(args.diagnostic_report)

        if recommendations:
            print("\n📊 Recommended fixes based on diagnostics:")
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = "🔴" if rec['priority'] == 'HIGH' else "🟡" if rec['priority'] == 'MEDIUM' else "🔵"
                print(f"  {priority_emoji} [{rec['priority']}] {rec['fix']}: {rec['reason']}")

            if args.apply_all:
                fixes_to_apply = [rec['fix'] for rec in recommendations]
        else:
            print("  ℹ️  No specific recommendations found")

    # Manual fix selection
    if args.fix_normalization:
        fixes_to_apply.append('layer_normalization')
    if args.fix_dropout:
        fixes_to_apply.append('modality_dropout')
    if args.fix_embeddings:
        fixes_to_apply.append('expose_embeddings')

    # If no fixes specified, show usage
    if not fixes_to_apply and not args.apply_all:
        print("\n⚠️  No fixes specified!")
        print("\nUsage:")
        print("  Auto-apply recommended fixes:")
        print("    python apply_fixes.py --diagnostic_report ./results/diagnostic_report.json \\")
        print("                          --model_file model.py --apply_all")
        print("\n  Apply specific fixes:")
        print("    python apply_fixes.py --model_file model.py \\")
        print("                          --fix_normalization --fix_dropout")
        print("\n  Preview changes:")
        print("    python apply_fixes.py --model_file model.py --fix_normalization --preview")
        return

    # Apply fixes
    print(f"\n📝 Applying {len(fixes_to_apply)} fixes...")

    success_count = 0
    for fix in fixes_to_apply:
        if fix == 'layer_normalization':
            if applicator.apply_fix_layer_normalization():
                success_count += 1
        elif fix == 'modality_dropout':
            if applicator.apply_fix_modality_dropout():
                success_count += 1
        elif fix == 'expose_embeddings':
            if applicator.apply_fix_expose_embeddings():
                success_count += 1
        elif fix == 'increase_graph_dim':
            applicator.apply_fix_increase_graph_dim()

    # Preview or save
    if args.preview:
        applicator.preview_changes()
        print("\n💡 Preview mode - no changes saved")
        print("   Remove --preview flag to apply changes")
    else:
        applicator.save(args.output_file)

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print(f"\n1. Review the changes in: {args.output_file}")
        print(f"2. Test the fixed model:")
        print(f"   - Update your training script to use the fixed model")
        print(f"   - Train and evaluate")
        print(f"3. Run diagnostics again to verify fixes worked:")
        print(f"   python run_diagnostics_v5.py --model_path new_model.h5 --quick")
        print(f"\n✅ Applied {success_count}/{len(fixes_to_apply)} fixes successfully")


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        print("\n" + "="*80)
        print("Multimodal Fusion Fix Applicator")
        print("="*80)
        print("\nAutomatically apply fixes to your model based on diagnostic results.")
        print("\nQuick start:")
        print("  1. Run diagnostics:")
        print("     python run_diagnostics_v5.py --model_path model.h5 --data test.pkl --num_batches 100")
        print("\n  2. Apply recommended fixes:")
        print("     python apply_fixes.py --diagnostic_report ./diagnostic_results/diagnostic_report.json \\")
        print("                           --model_file ../../kgcnn/literature/DenseGNN/_make_dense_multimodal_v5.py \\")
        print("                           --apply_all")
        print("\n  3. Retrain with fixed model")
        print("\nRun with --help for all options")
        print()
        sys.exit(0)

    main()
