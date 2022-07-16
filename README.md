zoom-setup
==========

Welcome	to the `zoom-setup`	tools. This	repository contains a collection of
python scripts to set up zoom simulations through the `ic_gen` IC generator,
in particular in a format suitable for use with	the public
[SWIFT](http://swift.dur.ac.uk) code.

Scripts are provided to
- Generate a mask for the zoom region from an existing simulation
- Create the unperturbed, multi-resolution "particle load" for `ic_gen`
- Convert the `ic_gen` output into a `SWIFT`-compatible single hdf5 file
- Configure a `SWIFT` parameter file with appropriate values for the ICs 

A concise overview of how to use these scripts is provided in `QUICKSTART.rst`.
For a more detailed description of features and available options, please see
the `/docs` subdirectory.

Parent simulation type
----------------------
Only `SWIFT` simulations with `VELOCIraptor` halo catalogues are supported as
input for generating masks. Support for `GADGET`/`SUBFIND` input has been dropped
in favour of better support for `SWIFT`.

Installation
------------
This repository provides top-level scripts, rather than an importable
package. Simply clone the repository to your local system with 
```bash
git clone https://github.com/ymbahe/zoom-setup.git
```

The following external libraries are required:
- `numpy`
- `pyyaml`
- `h5py`
- `scipy`
- `astropy`
- `cython`

You can ensure that they are installed by running
```bash
python3 -m pip install -r requirements.txt
```

Optionally, the following libraries are used:
- `mpi4py`, to enable parallel processing
- `matplotlib`, to generate diagnostic plots

These can be installed by running
```bash
python3 -m pip install -r requirements-opt.txt
```

**Note**: The particle load generator (sub-directory `particle_load`)
contains a module written in Cython for efficiency (`auxiliary_tools.pyx`).
This _should_ compile automatically through `pyximport`. If that fails for some
reason, please compile `auxiliary_tools.pyx` manually with
```bash
python3 setup.py build_ext --inplace
```
(The shorter `cythonize -i auxiliary_tools.pyx` tends to get stuck finding
the numpy headers, especially when running in virtual environments).
