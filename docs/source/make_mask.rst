.. _make_mask:

===============
Mask generation
===============

Overview
--------
The first step of the IC generation is to create a "mask" that defines which
part of the full simulation volume should be realized at the target resolution
(the rest will be filled with low-resolution boundary particles). This
mask region (in the very-high-redshift ICs) is constructed in such a way that
all particles within some region at lower redshift are high-resolution.
Typically, this region is centred on some target halo identified in a
previously run full-volume low-resolution version of the full "parent"
simulation that is to be re-simulated at higher resolution.

In the simplest case, the target halo is identified at z = 0 and a spherical
volume around the halo centre is to be refined. Alternatively, it is also
possible to refine an additional "padding" volume around this target region,
at multiple redshifts if desired. This can be helpful as a "safety buffer"
to prevent contamination of the target region with low-resolution particles
in the simulation run due to slight differences in particle trajectories
between parent and zoom simulations.

A more detailed description of how the mask is created can be found
:ref:`here <make_mask_method>`.

.. toctree::
   :hidden:

   make_mask_method

     
Instructions
------------
The mask generator is located in the ``make_mask`` directory. To run it, first
copy the template parameter file ``template_param_files/template.yml`` to a
location of your choice (e.g. ``param_files/suite_zoom_vr17.yml``). Then
adapt this copy as needed (see below for essential settings and all options).
Finally, generate the mask with

.. code-block:: bash

    $ python3 make_mask.py [your parameter file]

The script will then compute the mask and save it as an HDF5 file in the
location specified in the parameter file.

Parameter file
--------------
The parameter file is written in ```YAML`` format and contains a number of
options in the form of

.. code-block:: bash

    name:      value

The minimally required adjustments are as follows:

* ``snapshot_file``: The parent simulation snapshot (particle) file in which
  the target region should be selected.
* ``bits``: The number of bits in the Peano-Hilbert indexing of particle IDs
  used in the parent simulation.
* ``output_dir``: The directory in which to write the output (mask file and
  diagnostic plots).
* ``fname``: The name of the mask file to write (file name only, without
  extension).

In the standard case of centering the target region on a VELOCIraptor (VR)
halo, you also need to adjust

* ``vr_file``: The full path to the VR ``.properties`` file corresponding
  to the selected snapshot.
* ``group_number``: The index of the target VR halo
* ``highres_radius_r200``: The radius around the halo centre that should
  be resimulated at high resolution.

Note that some of these become optional when other values are specified
instead. A brief description of all possible settings is given in
``template.yml``


Mask file description
---------------------

The mask is saved as an HDF5 file with the following structure

::

    mask.hdf5
    ├── Params (group, attributes only)
    ├── Coordinates
    ├── FullMask
    └── TargetMask
        ├── Coordinates
	└── FullMask

* ``/Params`` is an empty group that contains, as attributes, every parameter
  in the parameter file (including implicitly set optional ones).
* ``/Coordinates`` contains the coordinates of the centres of all cells that
  are part of the mask (i.e. cells whose volume must be filled with high-
  resolution particles in the particle load). Coordinates are in Mpc, and
  are given relative to the mask centre. The dataset has four quantitative
  attributes:

  * ``bounding_length``: the full length of the longest side of the mask
    (i.e. the side length of a cube that contains the whole mask volume),
    in Mpc.
  * ``geo_centre``: the geometric centre of the mask within the parent
    simulation LCs, in Mpc.
  * ``grid_cell_width``: the side length of each cubic mask cell, in Mpc.
  * ``mask_corners``: The lower and upper vertices of a box (not necessarily
    a cube) that encloses all mask cells, in Mpc (relative to ``geo_centre``).

* ``/FullMask`` is a boolean 3D array with shape (n_x, n_y, n_z), the
  number of grid cells in each dimension from which the mask is selected.
  The array is ``True`` for cells that are part of the mask, and ``False`` for
  others. In addition to the four attributes of ``Coordinates``, which are
  copied for convenience, there are three additional attributes:

  * ``x_edges``: 1D array containing the edges of all cells along the x axis
    (in Mpc, relative to ``geo_centre``).
  * ``y_edges``: as ``x_edges``, but along the y axis.
  * ``z_edges``: as ``x_edges``, but along the z axis.

  The grid described by ``FullMask`` typically extends beyond the actual
  mask, so that the lowest (highest) values of ``x_edges``, ``y_edges``, and
  ``z_edges`` are generally below (above) those in ``mask_corners``.
  However, each cell marked as ``True`` corresponds exactly to one of the
  entries in ``Coordinates``.

* ``/TargetMask/Coordinates`` and ``TargetMask/FullMask`` are analogous to
  ``/Coordinates`` and ``/FullMask``, but for a special mask that only
  considers target particles and no padding. This is only output for possible
  diagnostic purposes, and not used anywhere in the particle load generation.
   



