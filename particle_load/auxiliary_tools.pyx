#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import sys
import ctypes
from numpy cimport ndarray, int32_t, float64_t, float32_t, int64_t
import numpy as np
cimport numpy as np


np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def assign_mask_cells(
    ndarray[float64_t, ndim=2] mask_pos,
    double mask_cell_width,
    ndarray[float64_t, ndim=2] gcell_pos,
    ndarray[int32_t, ndim=1] gcell_types,
    ):
    """
    Find local gcells that are part of the target-resolution region (Zone I).

    This is done by "painting" the mask cells onto the previously set up
    gcells. Every (local) gcell that overlaps with at least one mask cell is
    assigned type 0, the others type -1.

    All spatial quantities must be in units of the side length of one
    individual gcell, with the origin at the centre of the simulation box.

    Parameters
    ----------
    mask_pos : ndarray(float64) [N_mcells, 3]
        Coordinates of the centre of each mask cell
    mask_cell_width : double
        The side length of each mask cell.
    gcell_pos : ndarray(float64) [N_gcells, 3]
        Coordinates of the centre of each local gcall.
    gcell_types : ndarray(int32) [N_gcells]
        Type of each gcell (output only).

    Returns
    -------
    None (the output is stored in the input array `gcell_types`)

    Note
    ----
    In principle, it is possible to do this entirely via array operations.
    However, because each rank holds only a (generally non-contiguous) sub-set
    of gcells, the memory footprint is reduced significantly with the
    brute-force double loop approach taken here.

    Whether this is actually significant is another question (since the memory
    is dominated by particles, rather than gcells).

    """
    cdef int num_gcells = len(gcell_types)
    cdef long num_mask_cells = len(mask_pos)
    cdef long ii, jj
    cdef double half_mcell = mask_cell_width / 2.

    for ii in range(num_gcells):
        # Explicitly initialize each gcell, for clarity and robustness.
        gcell_types[ii] = -1

        for jj in range(num_mask_cells):
            if mask_pos[jj, 0] + half_mcell < gcell_pos[ii, 0]: continue
            if mask_pos[jj, 0] - half_mcell > gcell_pos[ii, 0] + 1.0: continue
            if mask_pos[jj, 1] + half_mcell < gcell_pos[ii, 1]: continue
            if mask_pos[jj, 1] - half_mcell > gcell_pos[ii, 1] + 1.0: continue
            if mask_pos[jj, 2] + half_mcell < gcell_pos[ii, 2]: continue
            if mask_pos[jj, 2] - half_mcell > gcell_pos[ii, 2] + 1.0: continue

            # If we get here, mask cell jj overlaps with gcell ii.
            gcell_types[ii] = 0
            break


def fill_gcells_with_particles(
    ndarray[float64_t, ndim=2] gcell_centres,
    ndarray[float64_t, ndim=2] coords_kernel,
    ndarray[float64_t, ndim=2] coords_parts,
    ndarray[float64_t, ndim=1] mass_parts,
    ndarray[float64_t, ndim=1] mass_kernel,
    long particle_offset
    ):
    """
    Fill the (local) gcells of one type with particles.

    Parameters
    ----------
    gcell_centres : ndarray(float) [N_cells_this, 3]
        The centres of the gcells to populate.
    coords_kernel : ndarray(float) [N_kernel, 3]
        The coordinate offsets w.r.t. the gcell centre of all particles with
        which to fill each gcell.
    coords_parts : ndarray(float) [N_cells, 3]
        The coordinates of all gcells, including those not processed now.
        This array is only used for output.
    mass_parts : ndarray(float) [N_cells]
        The masses of all gcells, including those not processed now.
        This array is only used for output.
    mass_kernel : float [N_in_kernel]
        The mass to assign to each particle in the kernel.
    part_offset : long
        The index of the first particle to generate here, in the full array.

    Returns
    -------
    None

    Notes
    -----
    Although it would be trivial to do this through plain numpy array
    operations, we use a direct loop approach here to avoid memory overheads
    (particle coordinates can be written directly to the output array).

    Examples
    --------
    >>> centres = np.array([-1., 2.5, 0.4])
    >>> kernel = np.array([[-0.1, -0.1, 0.], [0.2, 0., -0.2]])
    >>> coords = np.zeros((3, 3)) - 10
    >>> masses = np.zeros(3) - 10
    >>> m_kernel = np.array((1.0, 0.5))
    >>> fill_gcells_with_particles(centres, kernel, coords, m_kernel, 1)
    >>> print(coords)
    array([[-10., -10., -10.],
           [-1.1, 2.4, 0.4],
           [-0.8, 2.5, 0.2]])
    >>> print(masses)
    array([-10., 1.0, 0.5])

    """
    cdef long n_gcells = gcell_centres.shape[0]
    cdef long np_kernel = coords_kernel.shape[0]
    cdef long idx
    cdef long count = 0
    cdef Py_ssize_t ii, jj, kk, ind

    for ii in range(n_gcells):
        for jj in range(np_kernel):
            ind = particle_offset + count
            for kk in range(3):
                coords_parts[ind, kk] = (gcell_centres[ii, kk] +
                                 coords_kernel[jj, kk])
            mass_parts[ind] = mass_kernel[jj]
            count += 1


def fill_scube_layers(dict scube, dict nparts,
                      dict parts, int comm_rank, int comm_size):
    """
    Generate Zone III particles for the layers processed on local MPI rank.
    
    Parameters
    ----------
    scube : dict
        High-level properties of the scube to fill.
    nparts : dict
        Information about total number of particles to generate.
    parts : dict
        (Output) the structure containing the particle properties to fill.
    comm_rank : int
        The local MPI rank.
    comm_size : int
        The total number of MPI ranks.

    Returns
    -------
    None

    Note
    ----
    Throughout, the unit length is the simulation box side length (i.e. as
    in the final output). Recall that particle masses are in units of the
    total mass within the simulation box. 

    """
    cdef ndarray[float64_t, ndim=1, mode="c"] masses = parts['m']
    cdef ndarray[float64_t, ndim=2, mode="c"] pos = parts['pos']

    cdef double base_l_inner = scube['base_shell_l_inner']
    cdef int n_scells = scube['n_cells']
    cdef int n_extra = scube['n_extra']
    cdef int n_shells = scube['n_shells']
    cdef double l_ratio = scube['l_ratio']
    cdef double leap_mass = scube['leap_mass']
    cdef int i_max = n_scells - 1

    cdef int ishell, ix, iy, iz
    cdef double m
    cdef double l_inner, scell_size
    cdef int n_outer = 0
    cdef int n_all = 0

    # Initialize (local) index of current particle to generate
    cdef int p_idx = nparts['zone1_local'] + nparts['zone2_local']

    # Loop over shells inside out and add particles (if assigned to this rank)
    for ishell in range(n_shells):
        if ishell % comm_size != comm_rank: continue
        
        if ishell == n_shells - 1:
            n_scells += n_extra
            i_max = n_scells - 1

        l_inner = base_l_inner * l_ratio**ishell
        if l_inner > 1:
            raise ValueError(f"Shell {ishell}: l_inner = {l_inner:.3e} > 1!")

        scell_size = l_inner / (n_scells - 2)
        m = scell_size**3
        if ishell == n_shells - 1:
            m += leap_mass

        # Add the individual particles in this shell.
        # Looping over the full cube and only processing the outer shell turns
        # out to be most efficient for this. Similarly, specifying cells by
        # their doubled-symmetrized indices is more efficient than 0 -> i_max.
        for iz in range(-i_max, i_max+2, 2):
            for iy in range(-i_max, i_max+2, 2):
                for ix in range(-i_max, i_max+2, 2):
                    if max(abs(iz), abs(iy), abs(ix)) == i_max:
                        # All particles in the shell have the same mass
                        masses[p_idx] = m

                        # Equivalent to +/- (0.5*l_inner + 0.5*scell_size)                        
                        pos[p_idx, 0] = 0.5 * scell_size * ix
                        pos[p_idx, 1] = 0.5 * scell_size * iy
                        pos[p_idx, 2] = 0.5 * scell_size * iz
                        
                        if ishell == n_shells - 1:
                            n_outer += 1
                        p_idx += 1
                        n_all += 1

    print(f"Placed {n_all} particles, of which {n_outer} in outermost shell.")        
    np_target = nparts['tot_local']
    if p_idx != np_target:
        raise ValueError(
            f"Ended scube filling with {p_idx} particles, not {np_target}!")