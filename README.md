This is a Matrix Product State (MPS) algorithm for computing characters of irreducible representations of the symmetric group $S_n$. 
The algorithm computes an MPS encoding all characters of a given conjugacy class of $S_n$. It relies on a mapping from characters of $S_n$ to quantum spin chains  proposed by
[Marcos Crichigno and Anupam Prakash](https://arxiv.org/abs/2404.04322)

Example of how to use the algorithm can be found in **example.py**

Implementation of the MPS algorithm based on [mpnum library](https://mpnum.readthedocs.io/en/latest/) can be found in **character_builder.py**\
This version is optimized for speed.

Implementation of the MPS algorithm based on [quimb library](https://quimb.readthedocs.io/en/latest/) can be found in **character_builder_quimb.py**\
This version is not optimized.

See **experiments.ipynb** jupyter notebook to reproduce all the experiments from the paper. 

# Characters
CharacterBuilder (character_builder.py) - computes the character using MPS contraction and mpnum
CharacterBuilderQuimb (character_builder_quimb.py) - computes the character using MPS contraction and quimb

**Timing** 
timing_mps.py - runs the mpnum MPS algorithm experiments. Generates data in DATA/ 
timing_sage.py - runs the mpnum MPS algorithm experiments. Generates data in DATA/ 

**Plotting** 
make_plot - plots the character timing experiments 
make_pot_dim - plots 

