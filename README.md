#Molecule Expander

[WIP] Script based on RDkit. Will permute chemical fragments from a libary with fragments of a seed molecule to generate new smilar molecules.

seed.smi contains the seed molecule you want to expand around. mols.smi contains molecules that the script can look at for fragments to use in the expansion. All the SMILES are converted to vectors of Rdkit descriptors and the 10 k-nearest neighbors (in the descriptor space) to the seed SMILES out of the mols.smi SMILES are found. Script will then fragment the seed and 10 nearest neighbor SMILES and try to sub substitute the fragments of the seed SMILES with similar fragments from nearest neighbor SMILES.
