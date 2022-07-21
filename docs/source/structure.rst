.. _structure:

======================
Default file structure
======================

The zoom-setup scripts generate a number of output files, which are also
used as input for subsequent steps. The locations of these files are fully
flexible, but by default the structure below is used. The example simulation
suite here consists of two zoom regions (``VR17`` and ``VR42``), each of them
at multiple resolution levels (``res1``, ``res2``, and so on).

::

    suite
    ├── ICs
    │   ├── ic_gen
    │   │   ├── IC_Gen.x
    │   │   ├── [Power spectrum files]
    │   │   ├── VR17
    │   │   │   ├── mask_suite_VR17.hdf5
    │   │   │   ├── suite_VR17_res1
    │   │   │   │   ├── params.inp
    │   │   │   │   ├── particle_load
    │   │   │   │   │   └── fbinary
    │   │   │   │   │       ├── PL.0
    │   │   │   │   │       ├── [PL.1]
    │   │   │   │   │       └── ...
    │   │   │   │   │          
    │   │   │   │   ├── particle_load_info.hdf5
    │   │   │   │   ├── submit.sh
    │   │   │   │   └── [ic_gen output files]
    │   │   │   │
    │   │   │   ├── suite_VR17_res2
    │   │   │   │   └── ...
    │   │   │   │   
    │   │   │   └── suite_VR17_res3
    │   │   │       └── ...
    │   │   │   
    │   │   └── VR42
    │   │       └── ...
    │   │
    │   ├── ICs_suite_VR17_res1.hdf5
    │   ├── ICs_suite_VR17_res2.hdf5
    │   ├── ICs_suite_VR17_res3.hdf5
    │   │    
    │   ├── ICs_suite_VR42_res1.hdf5
    │   └── ...
    │   
    │   
    └── SIMULATION_RUNS
        ├── suite_VR17
        │   ├── suite_VR17_res1
        │   │   ├── params.yml
        │   │   ├── submit.sh
        │   │   ├── swift
        │   │   └── [other SWIFT run files]
        │   │
        │   ├── suite_VR17_res2
        │   │   └── ...
        │   │   
        │   └── suite_VR17_res3
        │
        └── suite_VR42
            └── ...
	    
The ``ICs`` directory contains the files used in creating the ICs (in the
``ic_gen`` subdirectory), and the finished SWIFT-compatible ICs files for
each individual region and resolution (``ICs_suite_VR17_res1.hdf5`` and so on).
Within the ``ic_gen`` subdirectory, each region has its own directory. It
contains the mask file (e.g. ``mask_suite_VR17.hdf5``), which is identical
across resolution levels, and then a separate directory for each resolution for
which ICs should be generated (``suite_VR17_res1``, etc.). Each of those
resolution-specific directories is used as the working directory for one
run of the `ic_gen` code.

``SIMULATION_RUNS`` is the directory for actually running the simulations
with SWIFT.


