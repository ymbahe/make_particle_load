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

[OLD BELOW]

As part of the particle load production, the `MakeGrid.pyx` is written in Cython and need to be compiled prior any use.
you can translate `MakeGrid.pyx` into native C code and compile it into a binary image using
```bash
cythonize -i MakeGrid.pyx
```
run in the same directory as `MakeGrid.pyx`. This operation produces the `MakeGrid.c` file, which encloses a C-translation
of the original Cython syntax and the `MakeGrid.cpython-*.so` shared object, which is the file that the Python interpreter
will actually point to when the `import` command is called. You may remove `MakeGrid.c` as it is only used by the Cython compiler 
to generate the shared object and is not needed by any other scripts at runtime.



Debug tips
-------
This section contains some useful debugging tips related to dependency configurations.

In order to check that you have the correct `PYTHONPATH` environment variable set-up and that all your dependencies are 
visible to the Python interpreter, you can print your current `PYTHONPATH` using
```python
import sys
print(sys.path)
```
The output contains a list of the directories, including the ones you have appended with `sys.path.append()`. We 
discourage the use of `os.environ['PYTHONPATH']`, as it may produce OS platform-dependent outputs, while its `sys` 
equivalent is platform-independent. If your custom directory appears in the `sys.path`, but the code exits with an
`ImportError`, check that the module is placed in the correct directory.
