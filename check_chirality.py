import os
import numpy as np
import time
from tqdm import tqdm
import pandas as pd
from Bio import PDB


def calculate_chirality(c_alpha, c_prev, n_next, c_sidechain):
    v1 = c_alpha - c_prev
    v2 = n_next - c_alpha
    v3 = c_sidechain - c_alpha

    # Cross product of vectors v1 and v2
    cross_prod = np.cross(v1, v2)

    # Dot product to determine orientation
    chirality = np.dot(cross_prod, v3)
    return 'L' if chirality > 0 else 'D'     # Return chirality: Positive for L, Negative for D


def evaluate_chirality(pdb_file, chain_name, echo=False):
    parser = PDB.PDBParser(QUIET=True)
    model = parser.get_structure('peptide', pdb_file)[0]
    num_res, num_chi = 0, 0
    for chain in model:
        if chain.id != chain_name:   # only compute for peptide
            continue
        for residue in chain:
            if PDB.is_aa(residue):
                try:
                    c_alpha = residue['CA'].get_vector().get_array()
                    c_prev = residue['C'].get_vector().get_array()
                    n_next = residue['N'].get_vector().get_array()
                    c_sidechain = residue['CB'].get_vector().get_array()
                    coords = c_alpha, c_prev, n_next, c_sidechain
                except KeyError:   # Handle cases where atoms might be missing
                    if echo: print(f"Error pass residue {residue.get_resname()}")
                    continue
                # print('...')
                if coords:
                    # print('asdvasdv')
                    chirality = calculate_chirality(*coords)
                    if echo: print(f"Residue {residue.get_resname()} {residue.id[1]}: {chirality}")
                    if chirality == 'D': num_chi += 1
                num_res += 1
            else:
                print(residue.id, PDB.is_aa(residue))
    return num_res, num_chi


if __name__ == '__main__':
    pdb_path = "./pep_output/dflow.pt_400_100_Trueflexible"
    save_path = f"./chirality_{time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())}.csv"
    pdb_name = os.listdir(pdb_path)

    chi_dict = {}
    for name in tqdm(pdb_name):
        if name.endswith('.csv'):
            continue
        folder_path = os.path.join(pdb_path, name)
        chi_dict[name] = {}
        for file_name in os.listdir(folder_path):
            num_res, num_chi = evaluate_chirality(os.path.join(folder_path, file_name), chain_name=name[-1])
            if num_res == 0:
                print(f'Warning: no residue in {os.path.join(folder_path, file_name)}')
                chi_dict[name][file_name] = 0.0
            else:
                chi_dict[name][file_name] = num_chi / num_res

    df = pd.DataFrame.from_dict({name: chi_dict[name] for name in chi_dict}, orient='index')
    df.to_csv(save_path)
    # path = './gt.pdb'
    # num_res, num_chi = evaluate_chirality(path, chain_name='B', echo=True)
    # print(num_res, num_chi)
