"""Multimodal DenseGNN V5 - Graph structure identical to original DenseGNN.

Uses GINELITE (same as original DenseGNN) + text fusion.
No global features, only node and edge Dense connections.
"""

import tensorflow as tf
from kgcnn.layers.geom import EuclideanNorm
from kgcnn.literature.DenseGNN._graph_network.graph_networks import CrystalInputBlock
from kgcnn.literature.DenseGNN._embedding_layers._atom_embedding import AtomEmbedding
from kgcnn.literature.DenseGNN._embedding_layers._edge_embedding import EdgeEmbedding
from kgcnn.crystal.periodic_table.periodic_table import PeriodicTable
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.layers.modules import LazyConcatenate
from kgcnn.layers.pooling import PoolingGlobalEdges
from ._gin_conv import GINELITE
from ._multimodal_fusion import ProjectionHead, GatedFusion, MiddleFusionModule
from copy import copy

ks = tf.keras


def get_features(x):
    if isinstance(x, dict):
        assert "features" in x.keys()
        return x["features"]
    return x


def make_model_multimodal_v5(
    inputs=None,
    name=None,
    input_block_cfg=None,
    input_embedding: dict = None,
    depth: int = None,
    gin_args: dict = None,
    gin_mlp: dict = None,
    output_mlp: dict = None,
    g_pooling_args: dict = None,
    # Multimodal config
    text_embedding_dim: int = 768,
    text_projection_dim: int = 128,
    graph_projection_dim: int = 128,
    use_middle_fusion: bool = True,
    middle_fusion_layers: list = None,
    middle_fusion_cfg: dict = None,
    late_fusion_type: str = "gated",
    late_fusion_cfg: dict = None,
):
    """Make multimodal DenseGNN V5 with graph structure identical to original DenseGNN.

    Key features:
        - Uses GINELITE (same as original DenseGNN)
        - Only node and edge Dense connections (no global)
        - No global/charge input
        - Text fusion via middle and late fusion
    """
    if middle_fusion_layers is None:
        middle_fusion_layers = [2]
    if middle_fusion_cfg is None:
        middle_fusion_cfg = {}
    if late_fusion_cfg is None:
        late_fusion_cfg = {'output_dim': 128, 'dropout': 0.1}
    if g_pooling_args is None:
        g_pooling_args = {'pooling_method': 'mean'}

    # === Graph Inputs (same as original DenseGNN) ===
    def in_inputs(key):
        return key in inputs and inputs[key] is not None

    edge_indices = ks.Input(**inputs['edge_indices'])
    atomic_number = ks.Input(**inputs['atomic_number'])
    edge_inputs, node_inputs = [], [atomic_number]

    if in_inputs('offset'):
        offset = ks.Input(**inputs['offset'])
        edge_inputs.append(offset)
    else:
        raise ValueError('Model needs "offset" input.')

    if in_inputs('voronoi_ridge_area'):
        inp_voronoi_ridge_area = ks.Input(**inputs['voronoi_ridge_area'])
        edge_inputs.append(inp_voronoi_ridge_area)

    if in_inputs('AGNIFinger'):
        inp_AGNIFinger = ks.Input(**inputs['AGNIFinger'])
        node_inputs.append(inp_AGNIFinger)

    # === Text Input ===
    text_spec = inputs.get('text_embedding')
    if text_spec:
        text_input = ks.Input(**text_spec)
    else:
        text_input = ks.Input(shape=(text_embedding_dim,), name='text_embedding')

    # === Process Graph (same as original DenseGNN) ===
    euclidean_norm = EuclideanNorm()
    distance = euclidean_norm(offset)

    if in_inputs('voronoi_ridge_area'):
        edge_input = (distance, inp_voronoi_ridge_area)
    else:
        edge_input = distance

    if in_inputs('AGNIFinger'):
        node_input = {'features': atomic_number, 'AGNIFinger': inp_AGNIFinger}
    else:
        node_input = {'features': atomic_number}

    # === Input Block (same as original DenseGNN, no global) ===
    crystal_input_block = GraphNetworkConfigurator.get_input_block(**input_block_cfg)
    edge_features, node_features, _, _ = crystal_input_block([
        edge_input, node_input, None, edge_indices
    ])

    n = get_features(node_features)
    ed = edge_features
    edi = edge_indices

    # Get node dimension from gin_mlp
    node_dim = gin_mlp.get('units', [64])[-1]

    # === Text Projection ===
    text_projection = ProjectionHead(
        embedding_dim=text_embedding_dim,
        projection_dim=text_projection_dim,
        dropout=0.1
    )
    text_emb = text_projection(text_input)

    # === Middle Fusion Modules ===
    node_fusion_modules = {}
    if use_middle_fusion:
        for layer_idx in middle_fusion_layers:
            node_fusion_modules[layer_idx] = MiddleFusionModule(
                node_dim=node_dim,
                text_dim=text_projection_dim,
                hidden_dim=middle_fusion_cfg.get('hidden_dim', node_dim * 2),
                dropout=middle_fusion_cfg.get('dropout', 0.1),
                num_heads=middle_fusion_cfg.get('num_heads', 0),
                use_gate_norm=middle_fusion_cfg.get('use_gate_norm', False),
                use_learnable_scale=middle_fusion_cfg.get('use_learnable_scale', False),
                initial_scale=middle_fusion_cfg.get('initial_scale', 1.0),
            )

    # === DenseGNN Layers (same as original: GINELITE, only node+edge) ===
    list_embeddings_n = [n]
    list_embeddings_e = [ed]

    for i in range(depth):
        if i > 0:
            n = GraphMLP(**gin_mlp)(n)
            ed = GraphMLP(**gin_mlp)(ed)

        # Middle fusion: inject text into node features
        if use_middle_fusion and i in middle_fusion_layers:
            n = node_fusion_modules[i]([n, text_emb])

        # GINELITE: same as original DenseGNN
        np, ep = GINELITE(**gin_args)([n, edi, ed])

        list_embeddings_n.append(np)
        list_embeddings_e.append(ep)

        n = LazyConcatenate()(list_embeddings_n)
        ed = LazyConcatenate()(list_embeddings_e)

    # === Output (same as original DenseGNN: edge pooling) ===
    graph_out = PoolingGlobalEdges(**g_pooling_args)(ep)

    # === Graph Projection ===
    graph_projection = ProjectionHead(
        embedding_dim=graph_out.shape[-1] if graph_out.shape[-1] else 128,
        projection_dim=graph_projection_dim,
        dropout=0.1
    )
    graph_emb = graph_projection(graph_out)

    # === Late Fusion ===
    if late_fusion_type == "gated":
        gated_fusion = GatedFusion(
            graph_dim=graph_projection_dim,
            text_dim=text_projection_dim,
            **late_fusion_cfg
        )
        fused = gated_fusion([graph_emb, text_emb])
        out = MLP(units=[late_fusion_cfg.get('output_dim', 128), 64, 1],
                  activation=['swish', 'swish', 'linear'])(fused)
    elif late_fusion_type == "concat":
        fused = tf.concat([graph_emb, text_emb], axis=-1)
        out = MLP(units=[128, 64, 1], activation=['swish', 'swish', 'linear'])(fused)
    else:
        # graph only (no text fusion at output)
        out = MLP(**output_mlp)(graph_out)

    # === Build Model ===
    input_list = edge_inputs + node_inputs + [edge_indices, text_input]
    return ks.Model(inputs=input_list, outputs=out, name=name)


class GraphNetworkConfigurator:
    """Configuration helper for input block."""

    @staticmethod
    def get_input_block(
        node_size=64, edge_size=64,
        atomic_mass=False, atomic_radius=False, electronegativity=False,
        ionization_energy=False, oxidation_states=False, melting_point=False,
        density=False, mendeleev=False, molarvolume=False,
        vanderwaals_radius=False, average_cationic_radius=False,
        average_anionic_radius=False, velocity_sound=False,
        thermal_conductivity=False, electrical_resistivity=False,
        rigidity_modulus=False,
        edge_embedding_args=None
    ):
        if edge_embedding_args is None:
            edge_embedding_args = {
                'bins_distance': 32, 'max_distance': 5., 'distance_log_base': 1.,
                'bins_voronoi_area': None, 'max_voronoi_area': None
            }

        periodic_table = PeriodicTable()

        atom_embedding_layer = AtomEmbedding(
            atomic_number_embedding_args={'input_dim': 119, 'output_dim': node_size},
            atomic_mass=periodic_table.get_atomic_mass() if atomic_mass else None,
            atomic_radius=periodic_table.get_atomic_radius() if atomic_radius else None,
            electronegativity=periodic_table.get_electronegativity() if electronegativity else None,
            ionization_energy=periodic_table.get_ionization_energy() if ionization_energy else None,
            oxidation_states=periodic_table.get_oxidation_states() if oxidation_states else None,
            melting_point=periodic_table.get_melting_point() if melting_point else None,
            density=periodic_table.get_density() if density else None,
            mendeleev=periodic_table.get_mendeleev() if mendeleev else None,
            molarvolume=periodic_table.get_molarvolume() if molarvolume else None,
            vanderwaals_radius=periodic_table.get_vanderwaals_radius() if vanderwaals_radius else None,
            average_cationic_radius=periodic_table.get_average_cationic_radius() if average_cationic_radius else None,
            average_anionic_radius=periodic_table.get_average_anionic_radius() if average_anionic_radius else None,
            velocity_sound=periodic_table.get_velocity_sound() if velocity_sound else None,
            thermal_conductivity=periodic_table.get_thermal_conductivity() if thermal_conductivity else None,
            electrical_resistivity=periodic_table.get_electrical_resistivity() if electrical_resistivity else None,
            rigidity_modulus=periodic_table.get_rigidity_modulus() if rigidity_modulus else None,
        )

        edge_embedding_layer = EdgeEmbedding(**edge_embedding_args)
        crystal_input_block = CrystalInputBlock(
            atom_embedding_layer,
            edge_embedding_layer,
            atom_mlp=MLP([node_size]),
            edge_mlp=MLP([edge_size])
        )
        return crystal_input_block
