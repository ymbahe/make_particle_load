.. _installation:

============
Installation
============

Obtaining a copy
----------------

The repository is available on github. You can obtain a fresh clone with

.. code-block:: bash

    $ git clone git@github.com:ymbahe/zoom-setup.git

An existing local repository can be updated with

.. code-block:: bash

    $ git pull


Required libraries
------------------

zoom-setup requires python3 (tested on versions 3.7 and later). It will not
work with python2.

In addition, the following external libraries are required:

* `astropy <https://www.astropy.org/>`_
* `cython <https://cython.org/>`_
* `h5py <https://www.h5py.org/>`_
* `numpy <https://numpy.org/>`_
* `pyyaml <https://pyyaml.org/>`_
* `scipy <https://scipy.org/>`_

You can ensure that they are installed by running

.. code-block:: bash

    $ python3 -m pip install -r requirements.txt

Optionally, the following libraries are used:

* `matplotlib <https://matplotlib.org/>`_ (to generate diagnostic plots)
* `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_
  (to enable parallel processing)

These can be installed by running

.. code-block:: bash

    $ python3 -m pip install -r requirements-opt.txt

Cython compilation
------------------

The :ref:`particle load generator <particle_load>` contains a module
written in Cython for efficiency (``auxiliary_tools.pyx``). In principle,
it should compile automatically through ``pyximport``, but occasionally
this fails. In this case, please compile ``auxiliary_tools.pyx`` manually
with

.. code-block:: bash

    $ python3 setup.py build_ext --inplace

As an alternative, you *might* be able to use

.. code-block:: bash

    $ cythonize -i auxiliary_tools.pyx

However, this has a tendency to get stuck finding the numpy headers, in
particular when running in virtual environments.

In either case, the manual compilation only has to be done once (and the
intermediate ``auxiliary_tools.c`` file can be deleted afterwards), unless
you make changes to ``auxiliary_tools.pyx``.
