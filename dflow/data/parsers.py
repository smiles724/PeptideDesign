"""Library for parsing different data structures."""
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB import Selection
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from easydict import EasyDict

from dflow.data import residue_constants, protein
from dflow.data.pep_constants import (AA, max_num_heavyatoms, restype_to_heavyatom_names, BBHeavyAtom)

Protein = protein.Protein


def _get_residue_heavyatom_info(res: Residue):
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)
    bfactor_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.float)
    restype = AA(res.get_resname())
    for idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
        if atom_name == '': continue
        if atom_name in res:
            pos_heavyatom[idx] = torch.tensor(res[atom_name].get_coord().tolist(), dtype=pos_heavyatom.dtype)
            mask_heavyatom[idx] = True
            bfactor_heavyatom[idx] = res[atom_name].get_bfactor()
    return pos_heavyatom, mask_heavyatom, bfactor_heavyatom


def parse_pdb(path, model_id=0, unknown_threshold=1.0, x_mirror=False):
    parser = PDBParser()
    structure = parser.get_structure(None, path)
    chains = Selection.unfold_entities(structure[model_id], 'C')
    chains.sort(key=lambda c: c.get_id())
    data = EasyDict(
        {'chain_id': [], 'chain_nb': [], 'resseq': [], 'icode': [], 'res_nb': [], 'aa': [], 'pos_heavyatom': [], 'mask_heavyatom': [], })
    tensor_types = {'chain_nb': torch.LongTensor, 'resseq': torch.LongTensor, 'res_nb': torch.LongTensor, 'aa': torch.LongTensor, 'pos_heavyatom': torch.stack,
                    'mask_heavyatom': torch.stack}
    count_aa, count_unk = 0, 0

    for i, chain in enumerate(chains):
        seq_this = 0  # Renumbering residues
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))  # Sort residues by resseq-icode
        for _, res in enumerate(residues):
            resname = res.get_resname()
            if not AA.is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue
            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK:   # UNKNOWN index: 20, not recorded
                count_unk += 1
                continue

            # Chain info
            data.chain_id.append(chain.get_id())
            data.chain_nb.append(i)

            # Residue types
            data.aa.append(restype)  # Will be automatically cast to torch.long

            # Heavy atoms
            pos_heavyatom, mask_heavyatom, bfactor_heavyatom = _get_residue_heavyatom_info(res)
            if x_mirror: pos_heavyatom[..., 0] *= -1  # mirror-image x-axis

            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)

            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]
            if seq_this == 0:
                seq_this = 1
            else:
                d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_CA_CA <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)

            data.resseq.append(resseq_this)
            data.icode.append(icode_this)
            data.res_nb.append(seq_this)

    if len(data.aa) == 0:
        return None, None

    if (count_unk / count_aa) >= unknown_threshold:
        return None, None

    seq_map = {}
    for i, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode)):
        seq_map[(chain_id, resseq, icode)] = i

    for key, convert_fn in tensor_types.items():
        data[key] = convert_fn(data[key])
    return data, seq_map


def process_chain(chain: Chain, chain_id: str) -> Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.
    
    Forked from alphafold.common.protein.from_pdb_string
    
    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.
    
    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.
    
    Took out lines 110-112 since that would mess up CDR numbering.
    
    Args:
        chain: Instance of Biopython's chain class.
    
    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        restype_idx = residue_constants.restype_order.get(res_shortname, residue_constants.restype_num)
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.
            res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return Protein(atom_positions=np.array(atom_positions), atom_mask=np.array(atom_mask), aatype=np.array(aatype), residue_index=np.array(residue_index),
                   chain_index=np.array(chain_ids), b_factors=np.array(b_factors))
