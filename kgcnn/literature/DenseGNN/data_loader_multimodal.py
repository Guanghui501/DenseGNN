"""Data loader for multimodal DenseGNN.

Example usage:
    from data_loader_multimodal import MultimodalDataLoader

    loader = MultimodalDataLoader(
        data_path='your_data.csv',
        target_col='formation_energy',
        text_col='description'
    )
    train_data, val_data, test_data = loader.get_datasets()
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class TextEncoder:
    """Encode text using MatSciBERT."""

    def __init__(self, model_name='m3rg-iitd/matscibert', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, model_max_length=self.max_length
            )
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to 768-dim vectors."""
        self._load_model()
        import torch

        encodings = self._tokenizer(
            texts, return_tensors='pt',
            padding=True, truncation=True, max_length=self.max_length
        )

        device = next(self._model.parameters()).device
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self._model(**encodings)
            cls_emb = outputs.last_hidden_state[:, 0, :]

        return cls_emb.cpu().numpy()


class MultimodalDataLoader:
    """Data loader for multimodal DenseGNN."""

    def __init__(
        self,
        data_path: str = None,
        data_df: pd.DataFrame = None,
        target_col: str = 'target',
        text_col: str = 'text',
        atoms_col: str = 'atoms',
        id_col: str = 'id',
        cif_dir: str = None,  # Directory containing CIF files
        cif_col: str = 'File_Name',  # Column with CIF filenames
        cutoff: float = 5.0,
        max_neighbors: int = 12,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        batch_size: int = 32,
        precompute_text_embeddings: bool = True
    ):
        """
        Args:
            data_path: Path to CSV/JSON file
            data_df: Or directly pass DataFrame
            target_col: Column name for target values
            text_col: Column name for text descriptions
            atoms_col: Column name for atomic structures (if atoms in DataFrame)
            id_col: Column name for sample IDs
            cif_dir: Directory containing CIF files (for generate_hse_csv.py output)
            cif_col: Column name for CIF filenames
            cutoff: Cutoff distance for neighbor search
            max_neighbors: Maximum number of neighbors
            train_ratio, val_ratio, test_ratio: Data split ratios
            random_seed: Random seed for reproducibility
            batch_size: Batch size for training
            precompute_text_embeddings: Whether to precompute text embeddings
        """
        self.target_col = target_col
        self.text_col = text_col
        self.atoms_col = atoms_col
        self.id_col = id_col
        self.cif_dir = cif_dir
        self.cif_col = cif_col
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.batch_size = batch_size

        # Load data
        if data_df is not None:
            self.df = data_df
        elif data_path is not None:
            self.df = self._load_data(data_path)
        else:
            raise ValueError("Must provide data_path or data_df")

        # Text encoder
        self.text_encoder = TextEncoder()
        self.text_embeddings = None

        if precompute_text_embeddings:
            self._precompute_text_embeddings()

    def _load_data(self, path: str) -> pd.DataFrame:
        """Load data from file."""
        path = Path(path)
        if path.suffix == '.csv':
            return pd.read_csv(path)
        elif path.suffix == '.json':
            return pd.read_json(path)
        elif path.suffix == '.pkl':
            return pd.read_pickle(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _precompute_text_embeddings(self):
        """Precompute text embeddings for all samples."""
        print("Precomputing text embeddings...")
        texts = self.df[self.text_col].tolist()

        # Encode in batches to avoid OOM
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_emb = self.text_encoder.encode(batch_texts)
            embeddings.append(batch_emb)

        self.text_embeddings = np.concatenate(embeddings, axis=0)
        print(f"Text embeddings shape: {self.text_embeddings.shape}")

    def _split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/val/test."""
        n = len(self.df)
        indices = np.arange(n)
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]

        return train_idx, val_idx, test_idx

    def _load_structure_from_cif(self, cif_filename: str):
        """Load structure from CIF file."""
        from pymatgen.core import Structure
        import os

        cif_path = os.path.join(self.cif_dir, cif_filename)
        return Structure.from_file(cif_path)

    def _atoms_to_graph_data(self, atoms_dict: Dict = None, cif_filename: str = None) -> Dict:
        """Convert atoms dict or CIF file to graph data for kgcnn.

        Args:
            atoms_dict: Dictionary with keys:
                - lattice_mat: 3x3 lattice matrix
                - coords: Nx3 fractional coordinates
                - elements: List of element symbols
                - cartesian: Whether coords are cartesian
            cif_filename: Or CIF filename (if cif_dir is set)

        Returns:
            Dictionary with graph data
        """
        from pymatgen.core import Structure, Lattice

        # Load structure from CIF or dict
        if cif_filename and self.cif_dir:
            structure = self._load_structure_from_cif(cif_filename)
        elif atoms_dict:
            lattice = Lattice(atoms_dict['lattice_mat'])
            coords = atoms_dict['coords']
            elements = atoms_dict['elements']
            structure = Structure(
                lattice, elements, coords,
                coords_are_cartesian=atoms_dict.get('cartesian', False)
            )
        else:
            raise ValueError("Must provide atoms_dict or cif_filename")

        # Get neighbors
        all_neighbors = structure.get_all_neighbors(self.cutoff)

        # Build edge list
        edge_indices = []
        offsets = []

        for i, neighbors in enumerate(all_neighbors):
            # Sort by distance and take top k
            neighbors = sorted(neighbors, key=lambda x: x[1])[:self.max_neighbors]
            for neighbor in neighbors:
                j = neighbor[2]  # neighbor index
                offset = neighbor[3]  # image offset
                edge_indices.append([i, j])
                # Calculate offset vector
                offset_vec = np.dot(offset, structure.lattice.matrix)
                offsets.append(offset_vec + structure[j].coords - structure[i].coords)

        # Get atomic numbers
        atomic_numbers = [site.specie.Z for site in structure]

        return {
            'atomic_number': np.array(atomic_numbers, dtype=np.int32),
            'edge_indices': np.array(edge_indices, dtype=np.int32),
            'offset': np.array(offsets, dtype=np.float32)
        }

    def _prepare_batch(self, indices: np.ndarray) -> Dict:
        """Prepare a batch of data."""
        batch_atomic_numbers = []
        batch_edge_indices = []
        batch_offsets = []
        batch_text_emb = []
        batch_targets = []

        # For ragged tensors
        node_splits = [0]
        edge_splits = [0]

        for idx in indices:
            row = self.df.iloc[idx]

            # Get graph data - from CIF file or atoms dict
            if self.cif_dir and self.cif_col in row:
                graph_data = self._atoms_to_graph_data(cif_filename=row[self.cif_col])
            else:
                atoms = row[self.atoms_col]
                if isinstance(atoms, str):
                    import json
                    atoms = json.loads(atoms)
                graph_data = self._atoms_to_graph_data(atoms_dict=atoms)

            n_nodes = len(graph_data['atomic_number'])
            n_edges = len(graph_data['edge_indices'])

            batch_atomic_numbers.append(graph_data['atomic_number'])
            batch_edge_indices.append(graph_data['edge_indices'] + node_splits[-1])
            batch_offsets.append(graph_data['offset'])

            node_splits.append(node_splits[-1] + n_nodes)
            edge_splits.append(edge_splits[-1] + n_edges)

            # Text embedding
            if self.text_embeddings is not None:
                batch_text_emb.append(self.text_embeddings[idx])
            else:
                text = row[self.text_col]
                emb = self.text_encoder.encode([text])[0]
                batch_text_emb.append(emb)

            # Target
            batch_targets.append(row[self.target_col])

        return {
            'atomic_number': np.concatenate(batch_atomic_numbers),
            'edge_indices': np.concatenate(batch_edge_indices),
            'offset': np.concatenate(batch_offsets),
            'text_embedding': np.array(batch_text_emb),
            'target': np.array(batch_targets, dtype=np.float32),
            'node_splits': np.array(node_splits[1:]),
            'edge_splits': np.array(edge_splits[1:])
        }

    def get_tf_dataset(self, indices: np.ndarray, shuffle: bool = True) -> tf.data.Dataset:
        """Create TensorFlow dataset."""
        def generator():
            idx_list = indices.copy()
            if shuffle:
                np.random.shuffle(idx_list)

            for i in range(0, len(idx_list), self.batch_size):
                batch_idx = idx_list[i:i+self.batch_size]
                yield self._prepare_batch(batch_idx)

        output_signature = {
            'atomic_number': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'edge_indices': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
            'offset': tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            'text_embedding': tf.TensorSpec(shape=(None, 768), dtype=tf.float32),
            'target': tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'node_splits': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'edge_splits': tf.TensorSpec(shape=(None,), dtype=tf.int32)
        }

        return tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    def get_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Get train/val/test datasets."""
        train_idx, val_idx, test_idx = self._split_data()

        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        train_ds = self.get_tf_dataset(train_idx, shuffle=True)
        val_ds = self.get_tf_dataset(val_idx, shuffle=False)
        test_ds = self.get_tf_dataset(test_idx, shuffle=False)

        return train_ds, val_ds, test_ds


# ============ Example Usage ============

def create_example_data():
    """Create example data for testing."""
    example_data = [
        {
            "id": "sample_001",
            "atoms": {
                "lattice_mat": [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
                "coords": [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
                "elements": ["Na", "Na", "Cl", "Cl"],
                "cartesian": False
            },
            "text": "Sodium chloride (NaCl) is an ionic compound with rock salt structure. It exhibits high melting point due to strong ionic bonds.",
            "target": -3.25
        },
        {
            "id": "sample_002",
            "atoms": {
                "lattice_mat": [[4.05, 0, 0], [0, 4.05, 0], [0, 0, 4.05]],
                "coords": [[0, 0, 0], [0.5, 0.5, 0.5]],
                "elements": ["Fe", "Fe"],
                "cartesian": False
            },
            "text": "Body-centered cubic iron (BCC Fe) is a ferromagnetic metal with high strength and ductility.",
            "target": -4.12
        }
    ]
    return pd.DataFrame(example_data)


def load_from_robocrys_csv(csv_path: str, cif_dir: str, batch_size: int = 32):
    """Load data from generate_hse_csv.py output.

    Args:
        csv_path: Path to CSV generated by generate_hse_csv.py
        cif_dir: Directory containing CIF files
        batch_size: Batch size

    Returns:
        MultimodalDataLoader instance

    Example:
        loader = load_from_robocrys_csv(
            csv_path='regression_data.csv',
            cif_dir='./cif_files/',
            batch_size=32
        )
        train_ds, val_ds, test_ds = loader.get_datasets()
    """
    return MultimodalDataLoader(
        data_path=csv_path,
        target_col='prop',
        text_col='Description',
        cif_dir=cif_dir,
        cif_col='File_Name',
        id_col='Id',
        batch_size=batch_size
    )


if __name__ == "__main__":
    # Example 1: Using generate_hse_csv.py output
    print("=" * 60)
    print("Example 1: Using robocrystallographer CSV + CIF files")
    print("=" * 60)
    print("""
    # Generate CSV first:
    python generate_hse_csv.py --cif_dir ./cifs --id_prop id_prop.csv --output data.csv

    # Then load:
    from data_loader_multimodal import load_from_robocrys_csv

    loader = load_from_robocrys_csv(
        csv_path='data.csv',
        cif_dir='./cifs/',
        batch_size=32
    )
    train_ds, val_ds, test_ds = loader.get_datasets()
    """)

    # Example 2: Using atoms dict directly
    print("\n" + "=" * 60)
    print("Example 2: Using atoms dict in DataFrame")
    print("=" * 60)

    df = create_example_data()
    print("\nExample DataFrame:")
    print(df[['id', 'text', 'target']].head())

    print("\nInitializing data loader...")
    loader = MultimodalDataLoader(
        data_df=df,
        target_col='target',
        text_col='text',
        atoms_col='atoms',
        batch_size=2,
        precompute_text_embeddings=True
    )

    print("\nGetting datasets...")
    train_ds, val_ds, test_ds = loader.get_datasets()

    print("\nSample batch:")
    for batch in train_ds.take(1):
        print(f"  atomic_number shape: {batch['atomic_number'].shape}")
        print(f"  edge_indices shape: {batch['edge_indices'].shape}")
        print(f"  offset shape: {batch['offset'].shape}")
        print(f"  text_embedding shape: {batch['text_embedding'].shape}")
        print(f"  target shape: {batch['target'].shape}")
