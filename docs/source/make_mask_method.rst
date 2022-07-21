.. _make_mask_method:

======================
Mask creation approach
======================

General idea
------------
The mask describes which part of the initial, unperturbed volume (also
referred to as "Lagrangian Coordinates"; LCs) should be realized at target
resolution during particle load generation. In principle, this mask region
can have an arbitrary shape; in practice, it is specified in terms of a finite
number of cubic cells. The fundamental objective of the mask generation is
deciding where to place these cells.

We want to set up the mask such that, once the zoom simulation is run, a
certain target region at some primary redshift (typically, z = 0) contains
only high-resolution (HR) particles -- i.e. particles originating from within
mask cells. Furthermore, we want these target HR particles to
always be shielded from lower-resolution boundary particles by a layer of
of "padding" HR particles, because undesirable artefacts tend to occur at the
edge of the HR region. In the simulation, there is no difference between
target and padding HR particles, but the distinction is useful during mask
creation.

Typically, the target region is a sphere of some radius centred on a particular
halo from a uniform low resolution "parent" simulation of the same volume.
But it is also possible to place the target region at an arbitrary position of
the parent simulation.

Identifying the mask region
---------------------------

To identify the volume in the LCs that evolves into the target region, we use
the particles of the low-resolution parent simulation. We can easily identify
all those particles that lie in the target region and find their position in
the LCs via their particle IDs. Because the particles only map points between
the (primary) snapshot and LCs, we need to treat a finite volume around each
particle as "flagged" (to be included in the mask). In analogy, we can
flag the region of the LCs whose particles end up occupying the padding zone
around the target region. This must also be included in the mask.

There are some subtleties that require additional care in setting up the mask:

* In detail, the mapping between LCs and any snapshot may differ slightly
  between parent and zoom simulation (in other words: there is no guarantee
  that two particles that originate from almost the same place in the LCs
  end up in almost the same place in the two simulations). We may therefore
  need some additional "safety" padding to account for this and make sure that
  the target region remains well padded in the zoom.
  
* On small scales, structure formation is not monotonic. Therefore, having the
  target region well padded at one snapshot (even z = 0) is no guarantee that
  (parts of) it may not get close to the high-resolution edge at another point
  in time. One may therefore want to track the target particles through
  multiple snapshots of the parent simulation, and identify the padding region
  in each of them. The final set of padding particles is then the union of
  all the sets of padding particles identified in each snapshot. One may
  also want to add padding directly in the LCs to avoid resolution artefacts
  at very high redshift.

Parameters for mask building
----------------------------

I/O parameters
^^^^^^^^^^^^^^

* ``output_dir``: The directory to which the output will be written.
* ``fname``: The name prefix for output files. The mask will be saved in
  ``output_dir`` as ``[fname].hdf5``.
* ``snapshot_file``: The location of the primary snapshot of the parent
  simulation from which the mask will be created. Can be omitted if the next
  two are specified.
* ``snapshot_base``: Template name for all relevant snapshots of the parent
  simulation. It must include a string ``$isnap`` as placeholder for the
  (4-digit, zero-padded) snapshot index.
* ``primary_snapshot``: Index of the primary snapshot in which target
  particles will be identified. Only relevant if ``snapshot_file`` is not
  provided.
* ``padding snaps``: [Optional] Snapshot(s) in which padding particles should
  be identified, in addition to the primary snapshot. This can be specified in
  three ways: (i) a single number for one snapshot; (ii) Two numbers separated
  by a space for a continuous snapshot range (e.g. ``0 5`` for snapshots 0
  to 5 inclusive); (iii) Three space-separated numbers for a sparse range
  (e.g. ``0 5 3`` for every third snapshot from 0 to 5 inclusive). If not
  specified, no snapshots beyond the primary are used for padding.

Padding options
^^^^^^^^^^^^^^^

* ``pad_lcs_as_particles``: [Optional] Set to ``True`` to identify padding
  particles in the LCs as in other snapshots (default: ``False``).
* ``highres_padding_width``: [Optional] Radius around the target region (or
  target particles) to treat as padding, in Mpc. Default: ``0``, no padding.
* ``cell_padding_width``: [Optional] Include cells in the LCs within this
  radius (in Mpc) around cells with target particles as padding. Default:
  ``0``, no cell padding.
* ``pad_full_cells``: [Optional] Pad all particles within cells that contain
  target particles, instead of only the target particles themselves.
  Generally not recommended (default: ``False``).

Mask generation
^^^^^^^^^^^^^^^

* ``min_num_per_cell``: [Optional] Minimum number of particles that must lie
  within a cell (in the LCs) for it to count as active. Setting this to > 1
  means that some cells that barely contribute to the target region or padding
  will not be included in the mask, but at the risk of stronger-than-wanted
  contamination of the target region by boundary particles. Default: 3.
* ``cell_size_mpc``: [Optional] Maximum acceptable size of mask cells in Mpc.
  Can also be set to ``auto``, in which case the value is calculated as a
  multiple of the mean inter-particle spacing (see next parameter). Default:
  ``3.0``.
* ``cell_size_mips``: Maximum acceptable size of mask cells in units of the
  mean inter-particle separation. Must be specified (and is only used) if
  ``cell_size_mpc == auto``. A value around 3 may be reasonable.
* ``dither_gap``: [Optional] If specified, the binning of particles to cells
  is performed 27 times instead of only once, with LC particle positions
  offset from their true values by all permutations of {-1, 0, 1} times
  this value in units of the mean inter-particle separation. In this way,
  a finite volume around each particle is selected, to reduce the risk of
  missing out relevant volume due to poor sampling. Default: ``None``, no
  dither applied.

Technical parameters
^^^^^^^^^^^^^^^^^^^^

* ``base_cosmology``: The name of the cosmology used for the parent simulation.
  This is only used to calculate the mean inter-particle separation.
* ``ids_are_ph``: Flag to indicate whether particle IDs in the parent
  simulation are Peano-Hilbert indices representing their position in the
  LCs. If not, a separate file must be provided that translates IDs into
  these Peano-Hilbert IDs.
* ``bits``: The number of bits that were used in computing the Peano-Hilbert
  IDs for the parent simulation (EAGLE runs use 14).
* ``data_type``: The type of snapshot from which the mask should be generated.
  Currently, ``swift`` (snapshot from a SWIFT simulation) is the only
  supported option.
* ``divide_ids_by_two``: Set this to ``True`` to divide particle IDs by 2
  before interpreting them as Peano-Hilbert indices (or looking them up).
  This is typically required if the parent run included gas particles.
* ``direct_primary_load``: Switch to load the entire primary snapshot (1)
  instead of only a subregion of it (0). Note that this affects only the
  primary snapshot; any other snapshots that may be required for padding
  are always read in full because particle positions within them are generally
  not known in advance.
* ``highres_diffusion_buffer``: If only a sub-region of the primary snapshot
  is loaded, expand it by this radius (in Mpc). This is because *only*
  particles loaded in the primary snapshot can be considered as padding
  particles, and some padding particles identified at higher redshift may
  have moved away from the target region by z ~ 0. Not used if
  ``direct_primary_load == 1``.

Mask regularization
^^^^^^^^^^^^^^^^^^^

* ``topology_fill_holes``: If ``True`` (default), holes within the full
  mask will be filled.
* ``topology_dilation_n_iter``: Number of iterations of the algorithm for
  extrusion (default: 0, i.e. disabled).
* ``topology_closing_n_iter``: Number of iterations of the algorithm for
  rounding edges of the mask (default: 0, i.e. disabled).

VR selection
^^^^^^^^^^^^

* ``select_from_vr``: Master switch to enable identification of target region
  based on a VELOCIraptor (VR) catalogue.
* ``vr_file``: The (full) path to the VR group catalogue, i.e. the
  ``.properties`` output file from VR. It can contain a ``$isnap`` string,
  which is replaced by the primary snapshot index to be consistent with
  the snapshot file.
* ``sort_type``: The type of halo mass to use for sorting/selecting haloes.
  Default is ``m200crit``, alternatives are ``m500crit`` and ``None`` (in
  which case, the unsorted order in the VR catalogue is used).
* ``group_number``: If ``sort_type`` is ``m200crit`` or ``m500crit``, the
  halo in position ``group_number`` in descending mass order is selected.
  If ``sort_type`` is ``None``, the halo with index ``group_number`` in the
  unsorted VR catalogue is selected.
* ``target_mass``: If specified, the halo with mass closest to this value
  (in M_Sun) in ``sort_type`` is selected, irrespective of its position
  within the mass-ranked list.
* ``highres_radius_r200``: [Optional] The minimum radius of the high-resolution
  target region around the halo centre, in units of r200. Default: 0.
* ``highres_radius_r500``: [Optional] As ``highres_radius_r200``, but in units
  of r500. Default: 0.
* ``highres_radius_min``: [Optional] Fixed minimum radius of the target
  high-resolution region in Mpc, independent of r200 or r500. Default: 0.

Manual target region selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``centre``: Triple of three floats that specify the centre of the target
  high-resolution region in the primary snapshot.
* ``shape``: Shape of the target region. Can be ``sphere``, ``cuboid``, or
  ``slab``.
* ``radius``: Radius of the target region; only used if ``shape == sphere``.
* ``dim``: Side lengths of target high-resolution region; only used if
  ``shape`` is ``cuboid`` or ``slab``.
  

Outline of mask building algorithm
----------------------------------
In brief, the algorithm proceeds as follows:

#. The parameter file is read and parsed (``MakeMask.read_param_file()``).

#. (The relevant part of) particles in the primary snapshot is read in.
   Target particles, and padding particles from the primary snapshot, are
   identified (``MakeMask.load_primary_ids()`` or
   ``MakeMask.load_primary_ids_direct()``).

#. The Lagrangian Coordinates of the loaded particles are computed from their
   IDs (``MakeMask.compute_ic_positions()``).

#. If applicable, additional padding particles are identified in other
   snapshots and/or in the LCs (``MakeMask.find_extra_pad_particles()``).

#. A box is constructed that surrounds all target and padding particles
   identified so far, with some padding. The mask will be constructed within
   this box (``MakeMask.compute_bounding_box()``).

#. The box is split into cubic cells. All cells that are near a target
   particle are tagged, and (separately) all that are near to a target or
   padding particle. The latter is the "full" mask (``Mask.__init__()``).

#. Optionally, all particles within cells tagged as hosting target particles
   are promoted to target particles, and their padding particles are found.
   This was added as a way to separate target and padding high-res particles
   in the zoom simulation. In practice, it tends to identify too many padding
   particles to be of use (``Mask.find_particles_in_active_cells()`` and
   ``Mask.find_extra_pad_particles()``).

#. The (full) mask is expanded to accommodate potential new padding particles,
   and optionally additional padding cells around any cells that host target
   particles (``Mask.expand()``).

#. Optionally, some shape regularization is applied to the full mask
   (``Mask.refine()``).

#. A sub-box is drawn around the active mask cells. The full mask is also
   re-centred to the geometric centre of this sub-box
   (``Mask.compute_active_box()`` and ``Mask.recenter()``).

#. The coordinates of active mask cells are written to an HDF5 file, together
   with metadata such as the cell size and mask center (``MakeMask.save()``
   and ``Mask.write()``).




  




