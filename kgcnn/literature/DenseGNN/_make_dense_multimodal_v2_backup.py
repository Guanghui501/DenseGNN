"""Multimodal DenseGNN with text-graph fusion - V2.

Based on make_model_asu with added multimodal fusion capabilities.
"""

import tensorflow as tf
from kgcnn.layers.geom import EuclideanNorm
from kgcnn.literature.DenseGNN._graph_network.graph_networks import GraphNetwork, CrystalInputBlock
from kgcnn.literature.DenseGNN._embedding_layers._atom_embedding import AtomEmbedding
from kgcnn.literature.DenseGNN._embedding_layers._edge_embedding import EdgeEmbedding
from kgcnn.crystal.periodic_table.periodic_table import PeriodicTable
from kgcnn.layers.mlp import MLP, GraphMLP
from tensorflow.keras.layers import GRUCell
from kgcnn.layers.modules import Dense, OptionalInputEmbedding, LazyConcatenate
from ._dense_gnn_conv import DenseGNN
from ._multimodal_fusion import ProjectionHead, GatedFusion, MiddleFusionModule
from copy import copy

ks = tf.keras


def get_features(x):
    if isinstance(x, dict):
        assert "features" in x.keys()
        return x["features"]
    return x


def update_features(x, v):
    if isinstance(x, dict):
        x_ = copy(x)
        x_["features"] = v
        return x_
    return v


def make_model_multimodal_v2(
    inputs=None,
    name=None,
    input_block_cfg=None,
    output_block_cfg=None,
    input_embedding: dict = None,
    depth: int = None,
    gin_args: dict = None,
    gin_mlp: dict = None,
    graph_mlp: dict = None,
    n_units: int = None,
    # Multimodal config
    text_embedding_dim: int = 768,
    text_projection_dim: int = 128,
    graph_projection_dim: int = 128,
    use_middle_fusion: bool = True,
    middle_fusion_layers: list = None,
    late_fusion_type: str = "gated",
    late_fusion_cfg: dict = None,
):
    """Make multimodal DenseGNN based on make_model_asu architecture.

    Args:
        inputs: Input configuration dict
        name: Model name
        input_block_cfg: Input block configuration
        output_block_cfg: Output block configuration (with attention)
        input_embedding: Embedding configuration for graph input
        depth: Number of GNN layers
        gin_args: DenseGNN layer arguments
        gin_mlp: MLP arguments for node/edge
        graph_mlp: MLP arguments for global features
        n_units: Hidden units for global features
        text_embedding_dim: Dimension of text embeddings
        text_projection_dim: Projection dimension for text
        graph_projection_dim: Projection dimension for graph
        use_middle_fusion: Whether to use middle fusion
        middle_fusion_layers: List of layer indices for middle fusion
        late_fusion_type: "gated" or "concat"
        late_fusion_cfg: GatedFusion config

    Returns:
        tf.keras.Model
    """
    if middle_fusion_layers is None:
        middle_fusion_layers = [2]
    if late_fusion_cfg is None:
        late_fusion_cfg = {'output_dim': 128, 'dropout': 0.1}

    # === Graph Inputs ===
    def in_inputs(key):
        return key in inputs and inputs[key] is not None

    edge_indices = ks.Input(**inputs['edge_indices'])
    atomic_number = ks.Input(**inputs['atomic_number'])
    edge_inputs, node_inputs, global_inputs = [], [atomic_number], []

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

    # === Global/Charge Input ===
    env_input = ks.Input(**inputs['charge'])
    global_inputs.append(env_input)
    uenv = OptionalInputEmbedding(
        **input_embedding['graph'],
        use_embedding=len(inputs['charge']['shape']) < 1
    )(env_input)
    global_input = Dense(n_units, use_bias=True, activation='relu')(uenv)

    # === Text Input ===
    text_spec = inputs.get('text_embedding')
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

    # === Input Block ===
    crystal_input_block = GraphNetworkConfigurator.get_input_block(**input_block_cfg)
    output_block = GraphNetworkConfigurator.get_gn_block(**output_block_cfg)

    edge_features, node_features, global_features, _ = crystal_input_block([
        edge_input, node_input, global_input, edge_indices
    ])

    n = get_features(node_features)
    ed = edge_features
    ud = global_features
    edi = edge_indices

    # === Text Projection ===
    text_projection = ProjectionHead(
        embedding_dim=text_embedding_dim,
        projection_dim=text_projection_dim,
        dropout=0.1
    )
    text_emb = text_projection(text_input)

    # === Middle Fusion Modules ===
    middle_fusion_modules = {}
    if use_middle_fusion:
        for layer_idx in middle_fusion_layers:
            middle_fusion_modules[layer_idx] = MiddleFusionModule(
                node_dim=n_units,
                text_dim=text_projection_dim,
                hidden_dim=n_units * 2,
                dropout=0.1
            )

    # === DenseGNN Layers ===
    list_embeddings_n = [n]
    list_embeddings_e = [ed]
    list_embeddings_u = [ud]

    for i in range(depth):
        if i > 0:
            n = GraphMLP(**gin_mlp)(n)
            ed = GraphMLP(**gin_mlp)(ed)
            ud = GraphMLP(**graph_mlp)(ud)

        # Middle fusion: inject text into global features using MiddleFusionModule
        if use_middle_fusion and i in middle_fusion_layers:
            ud = middle_fusion_modules[i]([ud, text_emb])

        np, ep, up = DenseGNN(**gin_args)([n, edi, ed, ud])

        list_embeddings_n.append(np)
        list_embeddings_e.append(ep)
        list_embeddings_u.append(up)

        n = LazyConcatenate()(list_embeddings_n)
        ed = LazyConcatenate()(list_embeddings_e)
        ud = LazyConcatenate()(list_embeddings_u)

    # === Output Block (with attention) ===
    nodes_new = update_features(node_features, np)
    x = [ep, nodes_new, up, edi]
    _, _, graph_out, _ = output_block(x)
    graph_out = output_block.get_features(graph_out)

    # === Graph Projection ===
    graph_projection = ProjectionHead(
        embedding_dim=graph_out.shape[-1] if graph_out.shape[-1] else 1,
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
    else:
        fused = tf.concat([graph_emb, text_emb], axis=-1)
        out = MLP(units=[128, 64, 1], activation=['swish', 'swish', 'linear'])(fused)

    # === Build Model ===
    input_list = edge_inputs + node_inputs + global_inputs + [edge_indices, text_input]
    return ks.Model(inputs=input_list, outputs=out, name=name)


class GraphNetworkConfigurator:
    """Configuration helper for graph network blocks."""

    @staticmethod
    def get_gn_block(
        edge_mlp=None, node_mlp=None, global_mlp=None,
        aggregate_edges_local="sum", aggregate_edges_global="sum", aggregate_nodes="sum",
        return_updated_edges=True, return_updated_nodes=True, return_updated_globals=True,
        edge_attention_mlp_local=None, edge_attention_mlp_global=None, node_attention_mlp=None,
        edge_gate=None, node_gate=None, global_gate=None,
        residual_node_update=False, residual_edge_update=False, residual_global_update=False,
        update_edges_input=None, update_nodes_input=None, update_global_input=None,
        multiplicity_readout=False
    ):
        if edge_gate == 'gru' and edge_mlp:
            edge_gate = GRUCell(edge_mlp['units'][-1])
        else:
            edge_gate = None

        if node_gate == 'gru' and node_mlp:
            node_gate = GRUCell(node_mlp['units'][-1])
        else:
            node_gate = None

        if global_gate == 'gru' and global_mlp:
            global_gate = GRUCell(global_mlp['units'][-1])
        else:
            global_gate = None

        edge_mlp_layer = MLP(**edge_mlp) if edge_mlp else None
        node_mlp_layer = MLP(**node_mlp) if node_mlp else None
        global_mlp_layer = MLP(**global_mlp) if global_mlp else None
        edge_attention_mlp_local_layer = MLP(**edge_attention_mlp_local) if edge_attention_mlp_local else None
        edge_attention_mlp_global_layer = MLP(**edge_attention_mlp_global) if edge_attention_mlp_global else None
        node_attention_mlp_layer = MLP(**node_attention_mlp) if node_attention_mlp else None

        if update_edges_input is None:
            update_edges_input = [True, True, True, False]
        if update_nodes_input is None:
            update_nodes_input = [True, False, False]
        if update_global_input is None:
            update_global_input = [False, True, False]

        block = GraphNetwork(
            edge_mlp_layer, node_mlp_layer, global_mlp_layer,
            aggregate_edges_local=aggregate_edges_local,
            aggregate_edges_global=aggregate_edges_global,
            aggregate_nodes=aggregate_nodes,
            return_updated_edges=return_updated_edges,
            return_updated_nodes=return_updated_nodes,
            return_updated_globals=return_updated_globals,
            edge_attention_mlp_local=edge_attention_mlp_local_layer,
            edge_attention_mlp_global=edge_attention_mlp_global_layer,
            node_attention_mlp=node_attention_mlp_layer,
            edge_gate=edge_gate, node_gate=node_gate, global_gate=global_gate,
            residual_edge_update=residual_edge_update,
            residual_node_update=residual_node_update,
            residual_global_update=residual_global_update,
            update_edges_input=update_edges_input,
            update_nodes_input=update_nodes_input,
            update_global_input=update_global_input
        )
        return block

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
