.. Character Builder documentation master file, created by
   sphinx-quickstart on Mon Jan 20 13:11:20 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Classical algorithms for characters of the symmetric group
===============================

This is a companion package to the paper "Classical and quantum algorithms for  characters of the symmetric group" by Sergey Bravyi, David Gosset, Vojtech Havlicek an Louis Schatzki. 
It contains the implementation of the classical algorithms for computing characters of the symmetric group Sn and Kostka numbers based on a tensor network contraction of Matrix Product States (MPS).

The algorithm computes an MPS encoding all characters of a given conjugacy class of Sn. It relies on a mapping from characters of Sn to quantum spin chains  proposed by
`Marcos Crichigno and Anupam Prakash <https://arxiv.org/abs/2404.04322>`_. 

Example of how to use the algorithm can be found in **example.py**

Implementation of the MPS algorithm based on `mpnum library <https://mpnum.readthedocs.io/en/latest/>`_ can be found in **character_building/character_builder.py**.
This version is optimized for speed.

Implementation of the MPS algorithm based on `quimb library <https://quimb.readthedocs.io/en/latest/>`_ can be found in **character_building/character_builder_quimb.py**.

See **experiments.ipynb** jupyter notebook to reproduce all the experiments from the paper. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   character_building
   utils
