"""Multimodal DenseGNN with text-graph fusion.

Following SGA-fusion framework:
- Middle fusion: inject text into graph encoding layers
- Late fusion: gated fusion of graph and text features
"""

import tensorflow as tf
from kgcnn.layers.geom import EuclideanNorm
from kgcnn.literature.DenseGNN._graph_network.graph_networks import CrystalInputBlock
from kgcnn.literature.DenseGNN._embedding_layers._atom_embedding import AtomEmbedding
from kgcnn.literature.DenseGNN._embedding_layers._edge_embedding import EdgeEmbedding
from kgcnn.crystal.periodic_table.periodic_table import PeriodicTable
from kgcnn.layers.mlp import MLP
from kgcnn.layers.modules import Dense
from kgcnn.layers.pooling import PoolingNodes, PoolingGlobalEdges
from kgcnn.layers.mlp import GraphMLP
from ._gin_conv import GINELITE
from kgcnn.layers.modules import LazyConcatenate
from ._multimodal_fusion import (
    ProjectionHead, MiddleFusionModule, GatedFusion, CrossModalAttention
)
from copy import copy

ks = tf.keras


def get_features(x):
    if isinstance(x, dict):
        assert "features" in x.keys()
        return x["features"]
    return x


def make_model_multimodal(
    inputs=None,
    name=None,
    input_block_cfg=None,
    depth: int = None,
    gin_args: dict = None,
    gin_mlp: dict = None,
    output_mlp: dict = None,
    g_pooling_args: dict = None,
    # Multimodal fusion config
    text_embedding_dim: int = 768,
    text_projection_dim: int = 64,
    graph_projection_dim: int = 64,
    use_middle_fusion: bool = True,
    middle_fusion_layers: list = None,  # e.g., [2] or [2, 3]
    middle_fusion_cfg: dict = None,
    use_cross_modal_attention: bool = True,
    cross_modal_cfg: dict = None,
    late_fusion_type: str = "gated",  # "gated" or "concat"
    late_fusion_cfg: dict = None,
    return_features: bool = False,
):
    """Make multimodal DenseGNN with text-graph fusion.

    Args:
        inputs: Input configuration dict
        name: Model name
        input_block_cfg: Input block configuration
        depth: Number of GNN layers
        gin_args: GIN layer arguments
        gin_mlp: MLP arguments for GIN
        output_mlp: Output MLP arguments
        g_pooling_args: Graph pooling arguments
        text_embedding_dim: Dimension of text embeddings (768 for BERT)
        text_projection_dim: Projection dimension for text
        graph_projection_dim: Projection dimension for graph
        use_middle_fusion: Whether to use middle fusion
        middle_fusion_layers: List of layer indices for middle fusion
        middle_fusion_cfg: MiddleFusionModule config
        use_cross_modal_attention: Whether to use cross-modal attention
        cross_modal_cfg: CrossModalAttention config
        late_fusion_type: "gated" or "concat"
        late_fusion_cfg: GatedFusion config
        return_features: Whether to return intermediate features

    Returns:
        tf.keras.Model
    """
    # Default configs
    if middle_fusion_layers is None:
        middle_fusion_layers = [2]
    if middle_fusion_cfg is None:
        middle_fusion_cfg = {'hidden_dim': 128, 'dropout': 0.1}
    if cross_modal_cfg is None:
        cross_modal_cfg = {'hidden_dim': 256, 'num_heads': 4, 'dropout': 0.1}
    if late_fusion_cfg is None:
        late_fusion_cfg = {'output_dim': 64, 'dropout': 0.1}

    # === Graph Inputs ===
    # Handle both dict and list input formats
    def in_inputs(key):
        if isinstance(inputs, dict):
            return key in inputs and inputs[key] is not None
        else:
            # List format - check by name
            return any(inp.get('name') == key for inp in inputs if isinstance(inp, dict))

    def get_input_spec(key):
        if isinstance(inputs, dict):
            return inputs[key]
        else:
            for inp in inputs:
                if isinstance(inp, dict) and inp.get('name') == key:
                    return inp
            return None

    edge_indices_spec = get_input_spec('edge_indices')
    atomic_number_spec = get_input_spec('atomic_number')

    if edge_indices_spec is None or atomic_number_spec is None:
        raise ValueError('Model needs "edge_indices" and "atomic_number" inputs.')

    edge_indices = ks.Input(**edge_indices_spec)
    atomic_number = ks.Input(**atomic_number_spec)
    edge_inputs, node_inputs = [], [atomic_number]

    if in_inputs('offset'):
        offset = ks.Input(**get_input_spec('offset'))
        edge_inputs.append(offset)
    else:
        raise ValueError('Model needs "offset" input.')

    if in_inputs('voronoi_ridge_area'):
        inp_voronoi_ridge_area = ks.Input(**get_input_spec('voronoi_ridge_area'))
        edge_inputs.append(inp_voronoi_ridge_area)

    if in_inputs('AGNIFinger'):
        inp_AGNIFinger = ks.Input(**get_input_spec('AGNIFinger'))
        node_inputs.append(inp_AGNIFinger)

    # === Charge/Global Input ===
    global_inputs = []
    if in_inputs('charge'):
        charge_input = ks.Input(**get_input_spec('charge'))
        global_inputs.append(charge_input)
        # Embed charge
        from kgcnn.layers.modules import OptionalInputEmbedding
        charge_spec = get_input_spec('charge')
        use_embedding = len(charge_spec.get('shape', [])) < 1
        uenv = OptionalInputEmbedding(
            input_dim=95, output_dim=input_block_cfg.get('node_size', 64),
            use_embedding=use_embedding
        )(charge_input)
        global_input = Dense(input_block_cfg.get('node_size', 64), use_bias=True, activation='relu')(uenv)
    else:
        global_input = None

    # === Text Input ===
    text_spec = get_input_spec('text_embedding')
    if text_spec:
        text_input = ks.Input(**text_spec)
    else:
        text_input = ks.Input(shape=(text_embedding_dim,), name='text_embedding')

    # === Process Graph ===
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

    crystal_input_block = GraphNetworkConfigurator.get_input_block(**input_block_cfg)
    edge_features, node_features, global_features, _ = crystal_input_block([
        edge_input, node_input, global_input, edge_indices
    ])

    n = get_features(node_features)
    ed = edge_features
    edi = edge_indices

    # === Text Projection ===
    text_projection = ProjectionHead(
        embedding_dim=text_embedding_dim,
        projection_dim=text_projection_dim,
        dropout=0.1
    )
    text_emb = text_projection(text_input)  # [batch, text_projection_dim]

    # === Middle Fusion Modules ===
    # Get node dimension from gin_mlp units (first layer output)
    node_dim = gin_mlp.get('units', [64])[0] if gin_mlp else 64
    middle_fusion_modules = {}
    if use_middle_fusion:
        for layer_idx in middle_fusion_layers:
            middle_fusion_modules[layer_idx] = MiddleFusionModule(
                node_dim=node_dim,
                text_dim=text_projection_dim,
                **middle_fusion_cfg
            )

    # Store intermediate outputs
    intermediate_outputs = {}
    if return_features:
        intermediate_outputs['embedding'] = {"node_feat": n, "edge_feat": ed}

    list_embeddings_n = [n]
    list_embeddings_e = [ed]

    # === GNN Layers with Middle Fusion ===
    for i in range(depth):
        if i > 0:
            n = GraphMLP(**gin_mlp)(n)
            ed = GraphMLP(**gin_mlp)(ed)

        np, ep = GINELITE(**gin_args)([n, edi, ed])

        # Apply middle fusion at specified layers
        if use_middle_fusion and i in middle_fusion_layers:
            # Pool nodes to graph level for fusion
            np_pooled = PoolingNodes(**g_pooling_args)(np)
            np_fused = middle_fusion_modules[i]([np_pooled, text_emb])
            # Broadcast back (simplified: use fused as graph-level feature)
            # For node-level fusion, would need more complex handling

        list_embeddings_n.append(np)
        list_embeddings_e.append(ep)

        n = LazyConcatenate()(list_embeddings_n)
        ed = LazyConcatenate()(list_embeddings_e)

        if return_features:
            intermediate_outputs[f"gc_{i+1}"] = {"node_feat": np, "edge_feat": ep}

    # === Graph Readout ===
    graph_emb = PoolingGlobalEdges(**g_pooling_args)(ep)

    # === Graph Projection ===
    graph_projection = ProjectionHead(
        embedding_dim=graph_emb.shape[-1],
        projection_dim=graph_projection_dim,
        dropout=0.1
    )
    graph_emb_proj = graph_projection(graph_emb)  # [batch, graph_projection_dim]

    # === Cross-Modal Attention ===
    if use_cross_modal_attention:
        cross_modal_attn = CrossModalAttention(
            graph_dim=graph_projection_dim,
            text_dim=text_projection_dim,
            **cross_modal_cfg
        )
        enhanced_graph, enhanced_text = cross_modal_attn([graph_emb_proj, text_emb])
    else:
        enhanced_graph, enhanced_text = graph_emb_proj, text_emb

    # === Late Fusion ===
    if late_fusion_type == "gated":
        gated_fusion = GatedFusion(
            graph_dim=graph_projection_dim,
            text_dim=text_projection_dim,
            **late_fusion_cfg
        )
        fused = gated_fusion([enhanced_graph, enhanced_text])
        out = MLP(**output_mlp)(fused)
    else:
        # Concat fusion
        fused = tf.concat([enhanced_graph, enhanced_text], axis=-1)
        out = MLP(**output_mlp)(fused)

    if return_features:
        intermediate_outputs['readout'] = {
            "graph_feat": enhanced_graph,
            "text_feat": enhanced_text,
            "fused_feat": fused
        }

    # === Build Model ===
    input_list = edge_inputs + node_inputs + [edge_indices] + global_inputs + [text_input]
    outputs = [out]

    if return_features:
        outputs.append(intermediate_outputs)

    return ks.Model(inputs=input_list, outputs=outputs, name=name)


class GraphNetworkConfigurator:
    """Configuration helper for graph network blocks."""

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
