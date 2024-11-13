"""pep-rec dataset"""
import logging
import math
import os
import pickle

import joblib
import lmdb
import torch
from Bio.PDB import PDBExceptions
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

from dflow.data.parsers import parse_pdb
from dflow.data.pep_constants import BBHeavyAtom, PAD_RESIDUE_INDEX
from dflow.data.torsion import get_torsion_angle

DEFAULT_PAD_VALUES = {'aa': PAD_RESIDUE_INDEX,  # 0-19, 20
                      'chain_id': ' ', 'icode': ' ', }

DEFAULT_NO_PADDING = {}


class PaddingCollate(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, no_padding=DEFAULT_NO_PADDING, eight=True, point2point=False):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight
        self.point2point = point2point

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([torch.ones([l], dtype=torch.bool), torch.zeros([n - l], dtype=torch.bool)], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys = self._get_common_keys(data_list)
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {k: self._pad_last(v, max_length, value=self._get_pad_value(k)) if k not in self.no_padding else v for k, v in data.items() if k in keys}
            data_padded['res_mask'] = (self._get_pad_mask(data[self.length_ref_key].size(0), max_length))
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)


def preprocess_structure(task, x_mirror=False):
    try:
        pdb_path = task['pdb_path']
        # pep
        # process peptide and find center of mass
        pep = parse_pdb(os.path.join(pdb_path, 'peptide.pdb'), x_mirror=x_mirror)[0]
        center = torch.sum(pep['pos_heavyatom'][pep['mask_heavyatom'][:, BBHeavyAtom.CA], BBHeavyAtom.CA], dim=0) / (torch.sum(pep['mask_heavyatom'][:, BBHeavyAtom.CA]) + 1e-8)
        pep['pos_heavyatom'] -= center[None, None, :]   # translate pep center
        pep['torsion_angle'], pep['torsion_angle_mask'] = get_torsion_angle(pep['pos_heavyatom'], pep['aa'])  # calc angles after translation
        if len(pep['aa']) < 3 or len(pep['aa']) > 25:   # ignore too short or too long pep
            raise ValueError('peptide length not in [3,25]')

        # rec
        rec = parse_pdb(os.path.join(pdb_path, 'pocket.pdb'), x_mirror=x_mirror)[0]
        rec['pos_heavyatom'] = rec['pos_heavyatom'] - center[None, None, :]     # translate pep center
        rec['torsion_angle'], rec['torsion_angle_mask'] = get_torsion_angle(rec['pos_heavyatom'], rec['aa'])  # calc angles after translation
        rec['chain_nb'] += 1

        # meta data
        data = {'id': task['id'], 'generate_mask': torch.cat([torch.zeros_like(rec['aa']), torch.ones_like(pep['aa'])], dim=0).bool()}
        for k in rec.keys():
            if isinstance(rec[k], torch.Tensor):
                data[k] = torch.cat([rec[k], pep[k]], dim=0)
            elif isinstance(rec[k], list):
                data[k] = rec[k] + pep[k]
            else:
                raise ValueError(f'Unknown type of {rec[k]}')
        return data
    except (PDBExceptions.PDBConstructionException, KeyError, ValueError, TypeError) as e:
        logging.warning('[{}] {}: {}'.format(task['id'], e.__class__.__name__, str(e)))
        return None


class PepDataset(Dataset):
    MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB

    def __init__(self, structure_dir="./Data/PepMerge/", dataset_dir="./Data/", name='pep', x_mirror=None, y_mirror=None, z_mirror=None, reset=False):
        super().__init__()
        self.structure_dir = structure_dir
        self.dataset_dir = dataset_dir
        self.name = name
        self.x_mirror = x_mirror
        self.y_mirror = y_mirror
        self.z_mirror = z_mirror
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset)

    @property
    def _cache_db_path(self):
        return os.path.join(self.dataset_dir, f'{self.name}_structure_{"x_" if self.x_mirror else ""}cache.lmdb')

    def _connect_db(self):
        self._close_db()
        self.db_conn = lmdb.open(self._cache_db_path, map_size=self.MAP_SIZE, create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with self.db_conn.begin() as txn:
            keys = [k.decode() for k in txn.cursor().iternext(values=False)]
            self.db_ids = keys

    def _close_db(self):
        if self.db_conn is not None:
            self.db_conn.close()
        self.db_conn = None
        self.db_ids = None

    def _load_structures(self, reset):
        todo_pdbs = []
        if reset:
            if os.path.exists(self._cache_db_path):
                os.remove(self._cache_db_path)
                lock_file = self._cache_db_path + "-lock"
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            self._close_db()

        if not os.path.exists(self._cache_db_path):
            all_pdbs = os.listdir(self.structure_dir)
            names_ = []
            with open('../names.txt', 'r') as f:  # testset   # TODO: build data folder structure
                for line in f:
                    names_.append(line.strip())
            if 'train' in self.name:
                all_pdbs = [pdb_fname for pdb_fname in all_pdbs if pdb_fname not in names_]
            elif 'test' in self.name:
                all_pdbs = [pdb_fname for pdb_fname in all_pdbs if pdb_fname in names_]
            elif 'all' in self.name:
                all_pdbs = all_pdbs
            else:
                raise ValueError('Data split can also accept: train/test/all.')

            todo_pdbs = all_pdbs

        if len(todo_pdbs) > 0 and not os.path.exists(self._cache_db_path):
            self._preprocess_structures(todo_pdbs)

    def _preprocess_structures(self, pdb_list):
        tasks = []
        for pdb_fname in pdb_list:
            pdb_path = os.path.join(self.structure_dir, pdb_fname)
            tasks.append({'id': pdb_fname, 'pdb_path': pdb_path})

        data_list = joblib.Parallel(n_jobs=max(joblib.cpu_count() // 2, 1), )(
            joblib.delayed(preprocess_structure)(task, self.x_mirror) for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess'))

        db_conn = lmdb.open(self._cache_db_path, map_size=self.MAP_SIZE, create=True, subdir=False, readonly=False, )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is not None:
                    ids.append(data['id'])
                    txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

    def __len__(self):
        self._connect_db()  # make sure db_ids is not None
        return len(self.db_ids)

    def __getitem__(self, index):
        self._connect_db()
        id = self.db_ids[index]
        with self.db_conn.begin() as txn:
            data = pickle.loads(txn.get(id.encode()))
        return data
