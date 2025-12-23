"""Multimodal fusion modules for DenseGNN.

Implements text-graph multimodal fusion following SGA-fusion framework:
- ProjectionHead: Maps embeddings to common space
- MiddleFusionModule: Mid-level gated fusion
- GatedFusion: Late fusion with learned weights
"""

import tensorflow as tf
from tensorflow.keras import layers


class ProjectionHead(layers.Layer):
    """Projection head for mapping embeddings to a common space."""

    def __init__(self, embedding_dim, projection_dim=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.projection = layers.Dense(self.projection_dim)
        self.gelu = layers.Activation('gelu')
        self.fc = layers.Dense(self.projection_dim)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.layer_norm = layers.LayerNormalization()
        super().build(input_shape)

    def call(self, x, training=None):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x, training=training)
        x = x + projected
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'projection_dim': self.projection_dim,
            'dropout': self.dropout_rate
        })
        return config


class MiddleFusionModule(layers.Layer):
    """Middle fusion module for injecting text information into graph encoding.

    Supports both gated fusion and multi-head attention fusion:
    - num_heads=0: gated fusion (original)
    - num_heads>0: multi-head cross-attention + gated fusion
    """

    def __init__(self, node_dim=64, text_dim=64, hidden_dim=128, dropout=0.1,
                 use_gate_norm=False, use_learnable_scale=False, initial_scale=1.0,
                 num_heads=0, **kwargs):
        super().__init__(**kwargs)
        self.node_dim = node_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.use_gate_norm = use_gate_norm
        self.use_learnable_scale = use_learnable_scale
        self.initial_scale = initial_scale
        self.num_heads = num_heads

    def build(self, input_shape):
        # Text transformation
        self.text_dense1 = layers.Dense(self.hidden_dim, activation='relu')
        self.text_dropout = layers.Dropout(self.dropout_rate)
        self.text_dense2 = layers.Dense(self.node_dim)

        # Multi-head attention (if enabled)
        if self.num_heads > 0:
            self.query_dense = layers.Dense(self.hidden_dim)
            self.key_dense = layers.Dense(self.hidden_dim)
            self.value_dense = layers.Dense(self.hidden_dim)
            self.attention_output = layers.Dense(self.node_dim)
            self.attention_dropout = layers.Dropout(self.dropout_rate)
            self.attention_norm = layers.LayerNormalization()

        # Gate mechanism
        self.gate_dense = layers.Dense(self.node_dim, activation='sigmoid')

        # Layer normalization
        self.layer_norm = layers.LayerNormalization()
        self.output_dropout = layers.Dropout(self.dropout_rate)

        if self.use_gate_norm:
            self.gate_norm = layers.LayerNormalization()

        if self.use_learnable_scale:
            self.text_scale = self.add_weight(
                name='text_scale', shape=(),
                initializer=tf.constant_initializer(self.initial_scale),
                trainable=True
            )
        else:
            self.text_scale = tf.constant(1.0, dtype=tf.float32)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: [node_feat, text_feat]
                - node_feat: [batch, N, node_dim] or RaggedTensor
                - text_feat: [batch, text_dim]
        Returns:
            Enhanced node features
        """
        node_feat, text_feat = inputs

        # Transform text features
        text_transformed = self.text_dense1(text_feat)
        text_transformed = self.text_dropout(text_transformed, training=training)
        text_transformed = self.text_dense2(text_transformed)  # [batch, node_dim]
        text_transformed = text_transformed * self.text_scale

        # Multi-head attention (if enabled)
        if self.num_heads > 0:
            text_transformed = self._apply_attention(node_feat, text_transformed, training)

        # Broadcast text to all nodes
        if isinstance(node_feat, tf.RaggedTensor):
            enhanced = self._process_ragged(node_feat, text_transformed, training)
        else:
            enhanced = self._process_dense(node_feat, text_transformed, training)

        enhanced = self.layer_norm(enhanced)
        enhanced = self.output_dropout(enhanced, training=training)
        return enhanced

    def _apply_attention(self, node_feat, text_feat, training):
        """Apply multi-head cross-attention: nodes attend to text."""
        # Get node features for attention
        if isinstance(node_feat, tf.RaggedTensor):
            # Pool nodes per sample for attention query
            node_pooled = tf.reduce_mean(node_feat, axis=1)  # [batch, node_dim]
        else:
            if len(node_feat.shape) == 3:
                node_pooled = tf.reduce_mean(node_feat, axis=1)
            else:
                node_pooled = node_feat

        # Q from nodes, K/V from text
        Q = self.query_dense(node_pooled)  # [batch, hidden_dim]
        K = self.key_dense(text_feat)      # [batch, hidden_dim]
        V = self.value_dense(text_feat)    # [batch, hidden_dim]

        # Reshape for multi-head attention
        batch_size = tf.shape(Q)[0]
        head_dim = self.hidden_dim // self.num_heads

        Q = tf.reshape(Q, [batch_size, self.num_heads, head_dim])
        K = tf.reshape(K, [batch_size, self.num_heads, head_dim])
        V = tf.reshape(V, [batch_size, self.num_heads, head_dim])

        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(head_dim, tf.float32))
        attention_scores = tf.einsum('bnh,bnh->bn', Q, K) / scale  # [batch, num_heads]
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.attention_dropout(attention_weights, training=training)

        # Apply attention
        context = tf.einsum('bn,bnh->bnh', attention_weights, V)
        context = tf.reshape(context, [batch_size, self.hidden_dim])

        # Output projection
        attended = self.attention_output(context)  # [batch, node_dim]
        attended = self.attention_norm(text_feat + attended)

        return attended

    def _process_ragged(self, node_feat, text_transformed, training):
        """Process RaggedTensor node features."""
        # Get flat values and row splits
        flat_nodes = node_feat.flat_values  # [total_nodes, node_dim]
        row_splits = node_feat.row_splits   # [batch + 1]

        # Compute number of nodes per sample
        row_lengths = node_feat.row_lengths()  # [batch]

        # Repeat text_transformed for each node in each sample
        text_repeated = tf.repeat(text_transformed, row_lengths, axis=0)

        # Compute gate
        gate_input = tf.concat([flat_nodes, text_repeated], axis=-1)
        if self.use_gate_norm:
            gate_input = self.gate_norm(gate_input)
        gate_values = self.gate_dense(gate_input)

        # Apply gated fusion
        enhanced_flat = flat_nodes + gate_values * text_repeated

        # Reconstruct RaggedTensor
        enhanced = tf.RaggedTensor.from_row_splits(
            values=enhanced_flat,
            row_splits=row_splits
        )
        return enhanced

    def _process_dense(self, node_feat, text_transformed, training):
        """Process dense tensor node features."""
        node_rank = len(node_feat.shape)
        if node_rank == 2:
            # Graph-level features [batch, node_dim]
            gate_input = tf.concat([node_feat, text_transformed], axis=-1)
            if self.use_gate_norm:
                gate_input = self.gate_norm(gate_input)
            gate_values = self.gate_dense(gate_input)
            enhanced = node_feat + gate_values * text_transformed
        else:
            # Node-level features [batch, N, node_dim]
            text_broadcasted = tf.expand_dims(text_transformed, 1)
            text_broadcasted = tf.broadcast_to(text_broadcasted, tf.shape(node_feat))

            gate_input = tf.concat([node_feat, text_broadcasted], axis=-1)
            if self.use_gate_norm:
                gate_input = self.gate_norm(gate_input)

            gate_values = self.gate_dense(gate_input)
            enhanced = node_feat + gate_values * text_broadcasted
        return enhanced

    def get_config(self):
        config = super().get_config()
        config.update({
            'node_dim': self.node_dim,
            'text_dim': self.text_dim,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_rate,
            'use_gate_norm': self.use_gate_norm,
            'use_learnable_scale': self.use_learnable_scale,
            'initial_scale': self.initial_scale,
            'num_heads': self.num_heads
        })
        return config


class GatedFusion(layers.Layer):
    """Gated fusion module - learns dynamic weights to balance two modalities.

    gate_g, gate_t = normalized sigmoid outputs
    fused = gate_g * graph_transformed + gate_t * text_transformed
    """

    def __init__(self, graph_dim=64, text_dim=64, output_dim=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        # Gate for graph
        self.gate_graph_dense1 = layers.Dense(self.graph_dim // 2, activation='relu')
        self.gate_graph_dropout = layers.Dropout(self.dropout_rate)
        self.gate_graph_dense2 = layers.Dense(1, activation='sigmoid')

        # Gate for text
        self.gate_text_dense1 = layers.Dense(self.text_dim // 2, activation='relu')
        self.gate_text_dropout = layers.Dropout(self.dropout_rate)
        self.gate_text_dense2 = layers.Dense(1, activation='sigmoid')

        # Feature transformation
        self.graph_transform = layers.Dense(self.output_dim)
        self.text_transform = layers.Dense(self.output_dim)

        # Fusion transformation
        self.fusion_dense = layers.Dense(self.output_dim)
        self.fusion_norm = layers.LayerNormalization()
        self.fusion_activation = layers.Activation('relu')
        self.fusion_dropout = layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: [graph_feat, text_feat]
                - graph_feat: [batch, graph_dim]
                - text_feat: [batch, text_dim]
        Returns:
            fused: [batch, output_dim]
        """
        graph_feat, text_feat = inputs

        # Compute gate weights
        gate_g = self.gate_graph_dense1(graph_feat)
        gate_g = self.gate_graph_dropout(gate_g, training=training)
        gate_g = self.gate_graph_dense2(gate_g)

        gate_t = self.gate_text_dense1(text_feat)
        gate_t = self.gate_text_dropout(gate_t, training=training)
        gate_t = self.gate_text_dense2(gate_t)

        # Normalize gates
        gate_sum = gate_g + gate_t + 1e-8
        gate_g = gate_g / gate_sum
        gate_t = gate_t / gate_sum

        # Transform features
        graph_transformed = self.graph_transform(graph_feat)
        text_transformed = self.text_transform(text_feat)

        # Gated fusion
        fused = gate_g * graph_transformed + gate_t * text_transformed

        # Final transformation
        fused = self.fusion_dense(fused)
        fused = self.fusion_norm(fused)
        fused = self.fusion_activation(fused)
        fused = self.fusion_dropout(fused, training=training)

        return fused

    def get_config(self):
        config = super().get_config()
        config.update({
            'graph_dim': self.graph_dim,
            'text_dim': self.text_dim,
            'output_dim': self.output_dim,
            'dropout': self.dropout_rate
        })
        return config


class CrossModalAttention(layers.Layer):
    """Cross-modal attention between graph and text features."""

    def __init__(self, graph_dim=64, text_dim=64, hidden_dim=256,
                 num_heads=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.graph_dim = graph_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.scale = (hidden_dim // num_heads) ** -0.5

    def build(self, input_shape):
        # Graph-to-Text attention
        self.g2t_query = layers.Dense(self.hidden_dim)
        self.g2t_key = layers.Dense(self.hidden_dim)
        self.g2t_value = layers.Dense(self.hidden_dim)

        # Text-to-Graph attention
        self.t2g_query = layers.Dense(self.hidden_dim)
        self.t2g_key = layers.Dense(self.hidden_dim)
        self.t2g_value = layers.Dense(self.hidden_dim)

        # Output projections
        self.graph_output = layers.Dense(self.graph_dim)
        self.text_output = layers.Dense(self.text_dim)

        self.dropout = layers.Dropout(self.dropout_rate)
        self.layer_norm_graph = layers.LayerNormalization()
        self.layer_norm_text = layers.LayerNormalization()

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: [graph_feat, text_feat]
        Returns:
            enhanced_graph, enhanced_text
        """
        graph_feat, text_feat = inputs

        # Add sequence dimension
        graph_seq = tf.expand_dims(graph_feat, 1)
        text_seq = tf.expand_dims(text_feat, 1)

        # Graph-to-Text Attention
        Q_g2t = self.g2t_query(graph_seq)
        K_g2t = self.g2t_key(text_seq)
        V_g2t = self.g2t_value(text_seq)

        attn_g2t = tf.matmul(Q_g2t, K_g2t, transpose_b=True) * self.scale
        attn_g2t = tf.nn.softmax(attn_g2t, axis=-1)
        attn_g2t = self.dropout(attn_g2t, training=training)
        context_g2t = tf.squeeze(self.graph_output(tf.matmul(attn_g2t, V_g2t)), axis=1)

        # Text-to-Graph Attention
        Q_t2g = self.t2g_query(text_seq)
        K_t2g = self.t2g_key(graph_seq)
        V_t2g = self.t2g_value(graph_seq)

        attn_t2g = tf.matmul(Q_t2g, K_t2g, transpose_b=True) * self.scale
        attn_t2g = tf.nn.softmax(attn_t2g, axis=-1)
        attn_t2g = self.dropout(attn_t2g, training=training)
        context_t2g = tf.squeeze(self.text_output(tf.matmul(attn_t2g, V_t2g)), axis=1)

        # Residual + LayerNorm
        enhanced_graph = self.layer_norm_graph(graph_feat + context_g2t)
        enhanced_text = self.layer_norm_text(text_feat + context_t2g)

        return enhanced_graph, enhanced_text

    def get_config(self):
        config = super().get_config()
        config.update({
            'graph_dim': self.graph_dim,
            'text_dim': self.text_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout_rate
        })
        return config
