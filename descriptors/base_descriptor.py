
from collections import defaultdict
from data.external.spartap_features import (PDB_SPARTAp_DataReader)

from abc import ABC, abstractmethod
import pickle
import torch
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from ase import Atoms
from tqdm import tqdm
from utils.process_data import load_struct, struct_to_dataframe
import concurrent.futures
import os
from pathlib import Path
import numpy as np
import multiprocessing

from data.const import DATA_FOLDER
# from data.external.spartap_features import (PDB_SPARTAp_DataReader)

class BaseDescriptor(ABC):

    def __init__(
            self,
            desc_name,
            data_path=DATA_FOLDER
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bmrb_pdf_file = f"{DATA_FOLDER}/bmrb_refs.csv"
        self.valid_seq_path = f"{DATA_FOLDER}/valid-exp-sequences.csv"

        self.desc_name = desc_name

        self.desc_path = Path(data_path) / desc_name
        os.makedirs(self.desc_path, exist_ok=True)


    def get_envs(self, data_folder:str, rmax=5.0, num_workers=10, af_model=0):
        try:
            with open(f'{DATA_FOLDER}/envs/envs_dict_{af_model}.pkl', 'rb') as f:
                envs = pickle.load(f)
            print(f"Loading saved environments from {DATA_FOLDER}/envs")
        except FileNotFoundError:
            entries = self._get_entries()

            multiprocessing.set_start_method('spawn', force=True)
            envs = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._process_entry, entry._asdict(), data_folder, rmax, af_model) for entry in entries]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing envs"):
                    envs.append(future.result())

            with open(f'{DATA_FOLDER}/envs/envs_dict_{af_model}.pkl', 'wb') as f:
                pickle.dump(envs, f)

            print(f"Saved envs dict in {DATA_FOLDER}/envs")
        return envs

    def _get_entries(self):
        try:
            valid_exp_sequences = pd.read_csv(self.valid_seq_path).drop(["Unnamed: 0"], axis=1).set_index("bmrb_id")
            bmrb_pdb_mapping = pd.read_csv(self.bmrb_pdf_file).drop(["Unnamed: 0"], axis=1).set_index("bmrb_id")
        except KeyError:
            valid_exp_sequences = pd.read_csv(self.valid_seq_path).set_index("bmrb_id")
            bmrb_pdb_mapping = pd.read_csv(self.bmrb_pdf_file).set_index("bmrb_id")

        valid_exp_sequences = valid_exp_sequences.join(bmrb_pdb_mapping, how="inner")
        return list(valid_exp_sequences.itertuples())

    def _process_entry(self, entry, data_folder:str, rmax:float=5.0, af_model = 0):
        """
        Process PDB entry and return the environments (for each residue)

        :return environments {dict: len(sequence)}
        """
        # bmrb_id, pdb_id = entry['Index'], entry['pdb_id']
        bmrb_id = entry["Index"]
        # print(f"Processing PDB ID: {pdb_id}")
        print(f"Processing BMRB ID: {bmrb_id}")
        # folder_name = f"{bmrb_id}_{pdb_id}"
        folder_name = f"{bmrb_id}"
        folder_path = Path(data_folder) / folder_name
        af_struct_filename = folder_path / f"af_0{af_model}.pdb"

        if not af_struct_filename.exists():
            # print(f"missing pdb id {pdb_id}")
            print(f"missing bmrb id {bmrb_id}")
            return

        structure = load_struct(af_struct_filename)
        structure_df = struct_to_dataframe(structure)

        # structure_df["pdb_id"] = pdb_id
        structure_df["bmrb_id"] = bmrb_id
        structure_df["af_file_name"] = af_struct_filename
        environments = self._extract_amino_acid_env(structure_df, rmax=rmax)
        return environments


    def _extract_amino_acid_env(self, df, rmax=5.0):
        """
        Extract environments per amino acid (residue) based on proximity (rmax)
        """
        X = np.vstack(df['coords'].values)
        neight = NearestNeighbors().fit(X)
        environments = {}

        for res_id, sub_df in df.groupby('res_id'):
            Y = np.vstack(sub_df['coords'].values)
            indices = neight.radius_neighbors(Y, radius=rmax, return_distance=False, sort_results=False)

            # Collect unique residues from neighbors
            res_indices = {df.iloc[idx]['res_id'] for idx in set(np.concatenate(indices))}
            environments[res_id] = df[df['res_id'].isin(res_indices)]

        return environments

    @abstractmethod
    def get_model(self, model:str):
        pass

    def _compute_atoms(self, env):
        env_coords = np.stack([env["coords"].iloc[i, ] for i in range(env.shape[0])])
        element_ids = env["element"].values
        atoms_obj_env = Atoms(symbols=element_ids, positions=env_coords)
        return atoms_obj_env

    def _get_residue_dict(self, descriptors, res_id, env):
        env = env.reset_index(drop=True)
        assert descriptors.shape[0] == env.shape[0]
        indices = env.index[(env["res_id"] == res_id)]
        res_dict = {
            'res_id': res_id,
            'name': env.loc[indices]['name'].unique()[0],
            'atom': env.loc[indices]['atom'].tolist(),
            'element': env.loc[indices]['element'].tolist(),
            'coords': env.loc[indices]['coords'],
            f'{self.desc_name}_descriptor': descriptors[indices].tolist()
        }
        return res_dict

    @abstractmethod
    def _calc_descriptors(self, model, atoms_obj_env, **kwargs):
        pass

    def _filter_exisiting_files(self, envs, atom_lst):
        filtered_envs = []
        for amino_acid_env in envs:
            # it's enough to check only the first env as the file is created only if processed all the atoms in the amino acid
            env = next(iter(amino_acid_env.items()))[1]
            try:
                bmrb_id, pdb_id = env["bmrb_id"].unique()[0], env["pdb_id"].unique()[0]
                filename = f"{bmrb_id}_{pdb_id}"
            except KeyError:
                bmrb_id = env["bmrb_id"].unique()[0]
                filename = f"{bmrb_id}"
            for atom in atom_lst:
                if not Path(f"{self.desc_path}/{atom}/{filename}.pkl").exists():
                    filtered_envs += [amino_acid_env]
                    break
        return filtered_envs


    def generate_descriptors(self, data_folder:str, model_name:str,
                             rmax:float=5.0,
                             num_workers:int=20, af_model=0, **kwargs):
        # get all the envs of all bmrbs
        all_envs = self.get_envs(data_folder, rmax=rmax, num_workers=num_workers, af_model=af_model)
        # all_envs = self._filter_exisiting_files(all_envs, atom_lst) if filter_existing_files else all_envs
        model = self.get_model(model_name)

        for envs in tqdm(all_envs, desc="Computing descriptors"):
            bmrb_dict = {f"{i}": [] for i in range(1,len(envs)+1)}

            # desc_af_model_path = f"{self.desc_path}/af_0{af_model}"
            desc_af_model_path = f"{self.desc_path}"
            os.makedirs(desc_af_model_path, exist_ok=True)

            bmrb_id = envs[1]["bmrb_id"][0]  # working on bmrb_id
            if f"{bmrb_id}.pkl" in list(os.listdir(desc_af_model_path)):
                print(f"Skipped {bmrb_id} because it already exists!")
                continue

            for i, (res_id, env) in enumerate(envs.items()):
                atoms_obj_env = self._compute_atoms(env)
                descriptors = self._calc_descriptors(model, atoms_obj_env, **kwargs) # for each atom get descriptor in the env

                res_dict = self._get_residue_dict(descriptors, res_id, env) # filter only atoms in res_id

                bmrb_dict[f"{res_id}"] = res_dict

            with open(f"{desc_af_model_path}/{bmrb_id}.pkl", "wb") as f:
                pickle.dump(bmrb_dict, f)
                f.close()

#
# from collections import defaultdict
# from data.external.spartap_features import (PDB_SPARTAp_DataReader)
#
# from abc import ABC, abstractmethod
# import pickle
# import torch
# from sklearn.neighbors import NearestNeighbors
# import pandas as pd
# from ase import Atoms
# from tqdm import tqdm
# from utils.process_data import load_struct, struct_to_dataframe
# import concurrent.futures
# import os
# from pathlib import Path
# import numpy as np
# import multiprocessing
#
# BACKBONE_ATOMS = {"N", "CA", "C", "O", "H", "H1", "HA", "HA1", "HA2", "HA3"}
#
#
# class BaseDescriptor(ABC):
#
#     def __init__(self, desc_name, bmrb_pdb_path="./data/bmrb-pdb-refs.csv", valid_seq_path="./data/valid-exp-sequences.csv"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.bmrb_pdf_file = bmrb_pdb_path
#         self.valid_seq_path = valid_seq_path
#
#         self.desc_path = desc_name
#         os.makedirs(self.desc_path, exist_ok=True)
#
#
#     def get_envs(self, data_folder:str, rmax=5.0, num_workers=10):
#         entries = self._get_entries()
#
#         multiprocessing.set_start_method('spawn', force=True)
#         envs = []
#         with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
#             futures = [executor.submit(self._process_entry, entry._asdict(), data_folder, rmax) for entry in entries]
#             for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing envs"):
#                 envs.append(future.result())
#         # list of env dictionaries and features
#         return envs
#
#     def _get_entries(self):
#         valid_exp_sequences = pd.read_csv(self.valid_seq_path).drop(["Unnamed: 0"], axis=1).set_index("bmrb_id")
#         bmrb_pdb_mapping = pd.read_csv(self.bmrb_pdf_file).drop(["Unnamed: 0"], axis=1).set_index("bmrb_id")
#         valid_exp_sequences = valid_exp_sequences.join(bmrb_pdb_mapping, how="inner")
#         return list(valid_exp_sequences.itertuples())
#
#     def _process_entry(self, entry, data_folder:str, rmax:float=5.0):
#         bmrb_id, pdb_id = entry['Index'], entry['pdb_id']
#         print(f"Processing PDB ID: {pdb_id}")
#         folder_name = f"{bmrb_id}_{pdb_id}"
#         folder_path = Path(data_folder) / folder_name
#         af_struct_filename = folder_path / f"af2_00_relaxed.pdb"
#
#         if not af_struct_filename.exists():
#             print(f"missing pdb id {pdb_id}")
#             return
#
#         structure = load_struct(af_struct_filename)
#         structure_df = struct_to_dataframe(structure)
#
#         structure_df["pdb_id"] = pdb_id
#         structure_df["bmrb_id"] = bmrb_id
#         structure_df["af_file_name"] = af_struct_filename
#         environments = self._extract_amino_acid_env(structure_df, rmax=rmax)
#         features = PDB_SPARTAp_DataReader().df_from_file_3res(str(af_struct_filename.absolute()))
#         return environments, features
#
#     def _extract_amino_acid_env(self, df, rmax=5.0):
#         """
#         Extract environments per amino acid based on proximity (rmax)
#         """
#         X = np.vstack(df['coords'].values)
#         neight = NearestNeighbors().fit(X)
#         environments = {}
#
#         for res_id, sub_df in df.groupby('res_id'):
#             Y = np.vstack(sub_df['coords'].values)
#             indices = neight.radius_neighbors(Y, radius=rmax, return_distance=False, sort_results=False)
#
#             # Collect unique residues from neighbors
#             res_indices = {df.iloc[idx]['res_id'] for idx in set(np.concatenate(indices))}
#             environments[res_id] = df[df['res_id'].isin(res_indices)]
#
#         return environments
#
#     @abstractmethod
#     def get_model(self, model:str):
#         pass
#
#     def _compute_atoms(self, env):
#         env_coords = np.stack([env["coords"].iloc[i, ] for i in range(env.shape[0])])
#         element_ids = env["element"].values
#         atoms_obj_env = Atoms(symbols=element_ids, positions=env_coords)
#         return atoms_obj_env
#
#     def _get_atom_descriptor(self, descriptors, res_id, env):
#         env = env.reset_index(drop=True)
#         assert descriptors.shape[0] == env.shape[0]
#         indices = env.index[env["res_id"] == res_id].tolist()
#         atoms = env.loc[indices, "atom"].tolist()
#         result = {atoms[i].rstrip('1'): descriptors[idx] for i, idx in enumerate(indices)}
#         return result
#
#
#     @abstractmethod
#     def _calc_descriptors(self, model, atoms_obj_env, **kwargs):
#         pass
#
#     def _filter_exisiting_files(self, envs):
#         filtered_envs = []
#         for amino_acid_env, features in envs:
#             # it's enough to check only the first env as the file is created only if processed all the atoms in the amino acid
#             env = next(iter(amino_acid_env.items()))[1]
#             bmrb_id, pdb_id = env["bmrb_id"].unique()[0], env["pdb_id"].unique()[0]
#             filename = f"{bmrb_id}_{pdb_id}"
#             for folder in os.listdir(self.desc_path):
#                 if not Path(f"{self.desc_path}/{folder}/{filename}.pkl").exists():
#                     filtered_envs += [(amino_acid_env, features)]
#                     break
#         return filtered_envs
#
#     def _get_empty_targets_dict(self, atoms):
#         empty_targets = {"name": np.nan, "secondary_structure":np.nan, "cs":{}, "rc_cs":{}}
#         for a in atoms:
#             empty_targets["cs"][a] = np.nan
#             empty_targets["rc_cs"][a] = np.nan
#         return empty_targets
#
#     def _get_targets(self, path, res_id, features, atoms):
#         df_target = pd.read_csv(path)
#         assert len(df_target) > 0
#         targets=self._get_empty_targets_dict(atoms)
#         res_df = df_target[df_target["res_id"]==res_id]
#         if len(res_df) > 0:
#             targets["name"] = res_df["name"].values[0]
#             ss = res_df["secondary_structure"].values[0]
#             targets["secondary_structure"] = ss if ss != "-" else np.nan
#             for atom in atoms:
#                 cs = res_df[atom].values[0] if atom in res_df.columns else "-"
#                 targets["cs"][atom] = cs if cs != "-" else np.nan
#                 res_features = features.query("RES_NUM==@res_id")
#                 targets["rc_cs"][atom] = res_features[f"RC_{atom}"].values[0] if f"RC_{atom}" in res_features.columns else np.nan
#         return targets
#
#     def _get_filtered_descriptor(self, descriptors, res_id, env):
#         current_res_atoms = env[env["res_id"] == res_id]
#         # check if not only backbone
#         if set(current_res_atoms["atom"]).issubset(BACKBONE_ATOMS):
#             return None
#         return descriptors[current_res_atoms.index]
#
#
#     def generate_descriptors(self, data_folder:str, model_name:str, rmax:float=5.0,
#                              num_workers:int=20, filter_exisiting_files=True, **kwargs):
#         envs = self.get_envs(data_folder, rmax=rmax, num_workers=num_workers)
#         envs = self._filter_exisiting_files(envs) if filter_exisiting_files else envs
#         model = self.get_model(model_name)
#
#         for amino_acid_env, features in tqdm(envs, desc="Computing descriptors"):
#             prot_atom_descriptors = defaultdict(list)
#             for i, (res_id, env) in enumerate(amino_acid_env.items()):
#                 atoms_obj_env = self._compute_atoms(env)
#                 descriptors = self._calc_descriptors(model, atoms_obj_env, **kwargs)
#                 descriptor_dict = self._get_atom_descriptor(descriptors, res_id, env)
#                 bmrb_id, pdb_id = env["bmrb_id"].unique()[0], env["pdb_id"].unique()[0]
#                 targets = self._get_targets(f"./data/targets/{bmrb_id}_{pdb_id}.csv", res_id, features, descriptor_dict.keys())
#                 for atom in descriptor_dict.keys():
#                     coords = env[(env["res_id"] == res_id) & (env["atom"] == atom)]["coords"]
#                     prot_atom_descriptors[atom].append(
#                         {
#                             "bmrb_id": bmrb_id,
#                             "pdb_id": pdb_id,
#                             "descriptor": descriptor_dict[atom],
#                             "name": targets["name"],
#                             "secondary_structure": targets["secondary_structure"],
#                             "cs": targets["cs"][atom],
#                             "rc_cs": targets["rc_cs"][atom],
#                             "res_id": res_id,
#                             "atom": atom,
#                             "coord": coords.values[0] if atom in env[env["res_id"]==res_id]["atom"].tolist() else np.nan
#                         })
#
#             for atom in descriptor_dict.keys():
#                 os.makedirs(Path(self.desc_path) / atom, exist_ok=True)
#                 filename = f"{bmrb_id}_{pdb_id}"
#                 with open(f"{self.desc_path}/{atom}/{filename}.pkl", "wb") as f:
#                     pickle.dump(prot_atom_descriptors[atom], f)
#                     f.close()
