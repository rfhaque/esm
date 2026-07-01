"""Tests for ESMFold2 input preparation (prepare_input)."""

import numpy as np
import pytest
import torch
from rdkit import Chem

from esm.models.esmfold2.prepare_input import (
    build_chains_from_input,
    compute_distogram_conditioning,
    compute_token_bonds,
)
from esm.models.esmfold2.types import (
    AtomPairDistanceConditioning,
    LigandInput,
    StructurePredictionInput,
)


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",  # benzene: 6 atoms, 6 bonds
        # The drug-like ligand from the SMILES-vs-CCD issue.
        "COC1=CC=C(N2C3=C(C(C(N)=O)=N2)CCN(C4=CC=C(N5CCCCC5=O)C=C4)C3=O)C=C1",
    ],
)
def test_smiles_ligand_bonds_match_molecular_graph(smiles: str):
    """SMILES ligand bonds must match the molecular graph, not a clique (#313)."""
    spi = StructurePredictionInput(sequences=[LigandInput(id="B", smiles=smiles)])
    chains, tokens, atoms = build_chains_from_input(spi, seed=0)
    token_bonds = compute_token_bonds(tokens, atoms, spi, chains)

    mol = Chem.MolFromSmiles(smiles)
    assert len(tokens) == mol.GetNumAtoms()
    n_edges = int(token_bonds.sum().item()) // 2  # symmetric matrix
    assert n_edges == mol.GetNumBonds()
    assert n_edges < len(tokens) * (len(tokens) - 1) // 2  # not a clique


def test_sparse_atom_pair_distance_conditioning_for_two_ligands():
    spi = StructurePredictionInput(
        sequences=[
            LigandInput(id="L1", smiles="CC"),
            LigandInput(id="L2", smiles="CO"),
        ],
        atom_pair_distance_conditioning=[
            AtomPairDistanceConditioning(
                chain_id1="L1",
                res_idx1=0,
                atom_idx1=1,
                chain_id2="L2",
                res_idx2=0,
                atom_idx2=0,
                distance=4.0,
            )
        ],
    )
    chains, tokens, atoms = build_chains_from_input(spi, seed=0)
    disto_cond, disto_cond_mask = compute_distogram_conditioning(
        spi, chains, tokens, atoms, torch.zeros(len(tokens), 3, dtype=torch.float32)
    )

    expected_bin = int(
        np.clip(np.digitize([4.0], np.linspace(2.0, 22.0, 64 + 1)[:-1]) - 1, 0, 63)[0]
    )
    assert int(disto_cond_mask.sum().item()) == 2
    assert disto_cond_mask[1, 2]
    assert disto_cond_mask[2, 1]
    assert int(disto_cond[1, 2].item()) == expected_bin
    assert int(disto_cond[2, 1].item()) == expected_bin
