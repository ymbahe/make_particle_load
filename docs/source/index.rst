.. zoom-setup documentation master file, created by
   sphinx-quickstart on Mon Jul 18 07:26:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

====================
The zoom-setup tools
====================

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   structure
   make_mask
   particle_load
   convert_ics
   setup_swift

.. |br| raw:: html

   <br />

This repository contains a collection of python scripts to set up zoom
simulations through the `ic_gen` IC generator, in particular in a format
suitable for use with the public SWIFT (http://swift.dur.ac.uk) code.

Setup
-----
* :ref:`Installation <installation>`  
* :ref:`Default file structure <structure>`

Individual scripts
------------------
* :ref:`Generate a zoom mask <make_mask>`
* :ref:`Create input files for ic_gen <particle_load>`
* :ref:`Convert ic_gen outputs to SWIFT-compatible hdf5 file <convert_ics>`
* :ref:`Set up the SWIFT simulation run <setup_swift>`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
