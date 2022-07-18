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
To be provided.


Code description
----------------
To be provided.
