import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, Recap
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.Chem import Descriptors

import pandas as pd
import itertools
from itertools import zip_longest
import copy
from scipy import spatial
from multiprocessing import Pool
from sklearn.cluster import AffinityPropagation

import os

#Below are all utiity functions used by main class Expand
def get_headings(Ipc=False):
    headings = [desc[0] for desc in Descriptors.descList]
    #May want to remove Ipc this descriptor has a large variance.
    if Ipc is False:
        headings.remove('Ipc')
    return headings

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def get_latent_vecs(mols, data_dir, file_name, num_procs=4):

    headings = get_headings(Ipc=True)
    num_lines = int(min(50000, len(mols)))
    n = int(len(mols) / num_lines)

    f1 = open(data_dir+'input_mols_filtered.csv', 'ab')
    file_name = data_dir + file_name
    f2 = open(file_name, 'ab')
    for i, group in enumerate(grouper(mols, num_lines)):
        if i == 0:
            smi_head = 'smiles'
            header = ','.join(headings)
        else:
            header = ''
            smi_head = ''
        print('Processing group {}/{}'.format(i, n))
        with Pool(processes=num_procs) as pool:
            res = pool.map(calc_descrs_for_smiles, group)
        #remove any failures
        res = [x for x in res if float('inf') not in x]
        #seperate smiles and vector
        smi = [x.pop(0) for x in res]
        np.savetxt(f1, smi, header=smi_head, fmt='%s', delimiter=',', comments='', newline='\n')
        np.savetxt(f2, res, header=header, fmt='%.18e', delimiter=',', comments='', newline='\n')
    f1.close()
    f2.close()

def calc_descrs_for_smiles(smi):
    failed = [smi]
    failed.extend([float('inf') for i in Descriptors.descList])

    if smi is None:
        return failed

    try:
    	m = Chem.MolFromSmiles(smi)
    except:
        print('RDkit could not pass smiles')
        return failed

    if m is None:
        print('RDkit could not pass smiles')
        return failed

    discp = [y(m) for x,y in Descriptors.descList]
    if float('nan') in discp:
        print('RDkit has returned a nan')
        return failed

    res = [smi]
    res.extend(discp)
    return res

def k_near_search(vector_file, lib_vec_file, num_neighbours=10):
    #Vector file is a file containing the compund we want to expand around
    #lib_vec_file is a libary of compunds in vector form which will be search for neighbours
    data = pd.read_csv(vector_file, header=0)
    # correct heading order
    data = data.reindex(columns=get_headings())
    alcdata = data.values[0]
    
    all_neigh_dist = []
    all_neigh_index = []
    chunksize = 100000
    for i, chunk in enumerate(pd.read_csv(lib_vec_file, chunksize=chunksize, header=0)):
        print('Evaluating chunk {} of lenght {}'.format(i, len(chunk)))
        # correct heading order
        chunk = chunk.reindex(columns=get_headings())
        traning_data = chunk.values
        tree = spatial.KDTree(traning_data)
        ans = tree.query(np.array(alcdata), k=10)
        all_neigh_dist.extend(ans[0])
        all_neigh_index.extend(ans[1]+(i*chunksize))
    return [x for _,x in sorted(zip(all_neigh_dist, all_neigh_index))]

def get_smiles_by_index(smi_file, idxs):
    data = pd.read_csv(smi_file, header=0).values
    # correct heading order
    smiles = []
    for i in idxs:
        smiles.append(str(data[i][0]))
    return smiles  

class Expand(object):
    def __init__(self, smi, lib_smiles, core_smi=None, lib_file=None, fam_sub=True):
        self.smi = Chem.MolFromSmiles(smi)
        if lib_file is None:
            self.lib = gen_lib(lib_smiles)
        else:
            self.lib = load_lib(lib_file)
            
        if not fam_sub:
            self.core = Expand.get_stripped_core(self, core_smi)
            self.expanded = Expand.add(self)
        else:
            main_fragments = Recap.RecapDecompose(self.smi).children.keys()
            all_fragments = copy.deepcopy(self.lib)
            all_fragments.extend(list(main_fragments))
            affin_matrix = Expand.build_affin_matrix(self, all_fragments)
            families = Expand.build_families(self, all_fragments, affin_matrix)
            self.lib = Expand.get_family_map(self, families, main_fragments)
            self.expanded = Expand.fam_add(self)
            for x in  self.expanded:
                print(x)
            
    def fam_add(self):
        all_mols = []
        for k, v in self.lib.items():
            patt = Chem.MolFromSmiles(k[1:])
            for frag in v:
                repl = Chem.MolFromSmiles(frag[1:])
                try:
                    #print(Chem.MolToSmiles(self.smi))
                    rms = AllChem.ReplaceSubstructs(self.smi, patt, repl)
                    all_mols.extend([Chem.MolToSmiles(x) for x in rms])
                except:
                    print('Encountered bad mol')
        return all_mols
        
    def add(self):
        patt = Chem.MolFromSmiles('*')
        smiles = Chem.MolToSmiles(self.core)
        print('Expanding {}'.format(smiles))
        num_replacments = smiles.count('*')
        all_mols = []
        for fragments in itertools.combinations(self.lib, num_replacments):
            current_gen = [self.core]
            next_gen = []
            for i, frag in enumerate(fragments):
                for m1 in current_gen:
                    repl = Chem.MolFromSmiles(frag[1:])
                    try:
                        next_gen = AllChem.ReplaceSubstructs(m1, patt, repl)
                    except:
                        print('Encountered bad mol')
                        break
                    for m2 in next_gen:
                        #Chem.SanitizeMol(m2)
                        if i == num_replacments-1:
                            all_mols.append(Chem.MolToSmiles(m2))
                current_gen = copy.deepcopy(next_gen)
        return all_mols
        
    def sub(self):
        all_mols = set()
        print('Reducing {}'.format(Chem.MolToSmiles(self.smi)))
        for fragment in self.lib:
            m = copy.deepcopy(self.smi)
            patt = Chem.MolFromSmarts(fragment[1:])
            rm = AllChem.DeleteSubstructs(m,patt)
            all_mols.add(Chem.MolToSmiles(rm))
        for x in all_mols:
            print(x)
            
    def build_affin_matrix(self, smiles):
        ms = [Chem.MolFromSmiles(x) for x in smiles]
        fps = [Chem.RDKFingerprint(x) for x in ms]
        matrix = []
        for x in fps:
            line = []
            for y in fps:
                line.append(DataStructs.FingerprintSimilarity(x ,y))
            matrix.append(line)
        matrix = np.array(matrix)
        return matrix
            
    def build_families(self, smiles, affin_matrix):
        cluster = AffinityPropagation()
        cls = cluster.fit_predict(affin_matrix)
        fam = {}
        for a, b in zip(smiles, cls):
            if b in fam:
                fam[b].add(a)
            else:
                fam[b] = set({a})
        return fam
    
    def get_family_map(self, families, main_fragments):
        family_map = {}
        for frag in main_fragments:
            for k, v in families.items():
                if frag in v:
                    family_map[frag] = k

        return {k: list(families[v]) for (k, v) in family_map.items()}
    
    def get_stripped_core(self, core_smi):
        if core_smi is None:
            print('No core provided generating core with Murcko Scaffold')
            core = MurckoScaffold.GetScaffoldForMol(self.smi)
            if Chem.MolToSmiles(core) == Chem.MolToSmiles(self.smi):
                print('Murcko Scaffold failed selecting largest fragment as core')
                hierarch = Recap.RecapDecompose(self.smi).children.keys() 
                tmp = Chem.MolFromSmiles(max(hierarch, key=len))
        else:
            core = Chem.MolFromSmiles(core_smi)
            tmp = Chem.ReplaceSidechains(self.smi, core)
            
        return tmp

def load_lib(file):
    data = pd.read_csv(file, header=None)
    data = [x[0] for x in data.values] 
    return data    
    
def gen_lib(smiles):
    mol_lib = [Chem.MolFromSmiles(x) for x in smiles]
    hierarch = [Recap.RecapDecompose(x).children.keys() for x in mol_lib]
    fragments = [j for i in hierarch for j in i]
    return fragments 

if __name__ == '__main__':
	if not os.path.isfile('./mols.smi'):
		raise ValueError('Require mols.smi file in this directory containing libary of smiles')
	if not os.path.isfile('./seed.smi'):
		raise ValueError('Require seed.smi file in this directory containing single smiles to use as seed')

	if not os.path.isfile('./vecs.csv'):
		mols = pd.read_csv('./mols.smi', header=0).values
		mols = [x[0] for x in mols]
		get_latent_vecs(mols, data_dir='./', file_name='./vecs.csv', num_procs=4)

	seed_mol = pd.read_csv('./seed.smi', header=0).values
	seed_mol = [x[0] for x in seed_mol]
	if not os.path.isfile('./seed.csv'):
		get_latent_vecs(mols, data_dir='./', file_name='./seed.csv', num_procs=4)

	top_vecs = k_near_search('./seed.csv', './vecs.csv')
	closest_smi = get_smiles_by_index('./mols.smi', top_vecs)
	all_mol = Expand(seed_mol[0], closest_smi, fam_sub=True)
	print(seed_mol[0])
	














