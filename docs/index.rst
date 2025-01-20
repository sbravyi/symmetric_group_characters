.. Character Builder documentation master file, created by
   sphinx-quickstart on Mon Jan 20 13:11:20 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Character Builder documentation
===============================

This is a Matrix Product State (MPS) algorithm for computing characters of irreducible representations of the symmetric group $S_n$ and Kostka numbers.
The algorithm computes an MPS encoding all characters of a given conjugacy class of $S_n$. It relies on a mapping from characters of $S_n$ to quantum spin chains  proposed by
`Marcos Crichigno and Anupam Prakash <https://arxiv.org/abs/2404.04322>`_. 

Example of how to use the algorithm can be found in **example.py**

Implementation of the MPS algorithm based on `mpnum library <https://mpnum.readthedocs.io/en/latest/>`_ can be found in **character_builder.py**.
This version is optimized for speed.

Implementation of the MPS algorithm based on `quimb library <https://quimb.readthedocs.io/en/latest/>`_ can be found in **character_builder_quimb.py**.
There are some optimization of this version. 

See **experiments.ipynb** jupyter notebook to reproduce all the experiments from the paper. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   character_building
