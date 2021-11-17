#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import sys
import ctypes
from numpy cimport ndarray, int32_t, float64_t, float32_t, int64_t
import numpy as np
cimport numpy as np


np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def make_grid(
    int n_cells_x, int n_cells_y, int n_cells_z, int comm_rank, int comm_size,
    int num_cells):
    cdef int max_n_cells = max(n_cells_x, n_cells_y, n_cells_z)

    # ** TODO ** Doing this in Cython via a loop is total overkill.
    # It should **definitely** be possible to do this via straightforward

    # Simple index array over max number of cells per dimension...
    cdef ndarray[float64_t, ndim=1, mode="c"] range_lo = np.array(
        np.arange(0, max_n_cells+1), dtype='f8')

    # Set up output arrays
    cdef ndarray[float64_t, ndim=2, mode="c"] offsets = np.zeros(
        (num_cells, 3), dtype='f8')
    cdef ndarray[int32_t, ndim=1, mode="c"] cell_nos = np.empty(
        num_cells, dtype='i4')

    cdef float lo_x, lo_y, lo_z
    cdef Py_ssize_t i
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t idx_count = 0

    cdef float off = max_n_cells/2.   # Cell coordinates of cube centre

    # Loop through all cells in order
    for iz in range(n_cells_z):
        for iy in range(n_cells_y):
            for ix in range(n_cells_x):

                # If this cell "belongs" to the current rank, do something...
                if count % comm_size == comm_rank:

                    # Register cell index relative to centre of cell cube
                    offsets[idx_count, 0] = float(ix) - off
                    offsets[idx_count, 1] = float(iy) - off
                    offsets[idx_count, 2] = float(iz) - off
                    
                    # Register the (wrapped) scalar index of current cell
                    cell_nos[idx_count] = count
                    idx_count += 1
                count += 1

    if idx_count != num_cells:
        raise ValueError("I did not make the right number of cells!")

    offsets[:, 0] += (max_n_cells - n_cells_x) / 2.
    offsets[:, 1] += (max_n_cells - n_cells_y) / 2.
    offsets[:, 2] += (max_n_cells - n_cells_z) / 2.

    return offsets, cell_nos

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

    In principle, it is possible to do this entirely via array operations.
    However, because each rank holds only a (generally non-contiguous) sub-set
    of gcells, the memory footprint is reduced significantly with the
    brute-force double loop approach taken here.

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


def find_skin_cells(
    ndarray[int32_t, ndim=1] cell_types,
    ndarray[int32_t, ndim=1] cell_nos,
    int L, int cell_type
    ):
    """
    Find the neighbouring 6 skin cells of each cell in a list.

    Parameters
    ----------
    cell_types : ndarray(int32t) [N_gcells]
        The types of all local gcells.
    cell_nos : ndarray(int32t) [N_gcells]
        Some kind of indices of the local gcells. To be continued...
    L : int
        Some badly named parameter.
    cell_type : int
        The cell type to focus on...?

    Returns
    -------
    skin_cells : ndarray(int32t)
        The indices of the gcells tagged as skin neighbours.
    """

    cdef long i, this_cell_no, this_idx
    cdef long count = 0
    cdef skin_cells = np.ones(6*len(cell_nos), dtype='i4') * -1

    cdef ind_seed_gcells = np.where(cell_types == cell_type)[0]

    if len(ind_seed_gcells) == 0:
        return []
    else:
        for this_idx in ind_seed_gcells:
            this_cell_no = cell_nos[this_idx]

            if this_cell_no % L != 0:
                skin_cells[count] = this_cell_no - 1
            count += 1
            if this_cell_no % L != L - 1:
                skin_cells[count] = this_cell_no + 1
            count += 1
            if ((this_cell_no / L) % L) != 0:
                skin_cells[count] = this_cell_no - L
            count += 1
            if ((this_cell_no / L) % L) != L - 1:
                skin_cells[count] = this_cell_no + L
            count += 1
            if (this_cell_no / L**2) % L != L - 1:
                skin_cells[count] = this_cell_no + L**2 
            count += 1
            if (this_cell_no / L**2) % L != 0:
                skin_cells[count] = this_cell_no - L**2
            count += 1

        # Fish out actually used array entries (we allocated the max possible)
        ind_assigned = np.nonzero(skin_cells != -1)[0]
        if len(ind_assigned) == 0:
            return []
        else:
            return np.unique(skin_cells[ind_assigned])

def gen_layered_particles_slab(double slab_width, double boxsize, int nq, int nlev, double dv,
        int comm_rank, int comm_size, int n_tot_lo, int n_tot_hi,
        ndarray[float64_t, ndim=1, mode="c"] coords_x, ndarray[float64_t, ndim=1, mode="c"] coords_y,
        ndarray[float64_t, ndim=1, mode="c"] coords_z, ndarray[float64_t, ndim=1, mode="c"] masses,
        int nq_reduce, int extra):
    """
    Generate Zone III particles for the layers processed on local rank.
    """

    cdef double offset, m_int_sep
    cdef int i, j, s, l, idx, this_nq
    cdef double half_slab = slab_width / 2.
    cdef long count = 0

    # Loop over each level.
    offset = half_slab
    count = 0
    for l in range(nlev):

        if l == nlev-1: nq += extra
        
        # Mean interparticle sep at this level.
        m_int_sep = boxsize/float(nq)

        if l % comm_size == comm_rank:
            for s in range(2):              # Both sides of the slab.
                for j in range(nq):
                    for i in range(nq):
                        idx = n_tot_hi + count
                        if l == nlev-1:
                            masses[idx] = m_int_sep**3.+dv
                        else:
                            masses[idx] = m_int_sep**3.
                
                        coords_x[idx] = i*m_int_sep + 0.5*m_int_sep
                        coords_y[idx] = j*m_int_sep + 0.5*m_int_sep
                        if s == 0:
                            coords_z[idx] = boxsize/2. + 0.5*m_int_sep + offset
                        else:
                            coords_z[idx] = boxsize/2. - 0.5*m_int_sep - offset
                        count += 1
        offset += m_int_sep
        nq -= nq_reduce

    assert count == n_tot_lo, 'Rank %i Slab outer particles dont add up %i != %i (nq_reduce=%i extra=%i finishing_nq=%i nlev=%i)'\
            %(comm_rank, count, n_tot_lo, nq_reduce, extra, nq+nq_reduce, nlev)
    if comm_rank == 0: print('Generated %i outer slab layers.'%nlev)
    coords_x[n_tot_hi:] /= boxsize
    coords_x[n_tot_hi:] -= 0.5
    coords_y[n_tot_hi:] /= boxsize
    coords_y[n_tot_hi:] -= 0.5
    coords_z[n_tot_hi:] /= boxsize
    coords_z[n_tot_hi:] -= 0.5
    masses[n_tot_hi:] /= boxsize**3.

def guess_nq(double lbox, int nq, int extra, int comm_rank, int comm_size):
    """
    Do something suitably obscure.

    Parameters
    ----------
    lbox : double
        The simulation box sidelength in units of the gcube sidelength.
    nq : int
        The nq value to try out
    extra : int
        ???
    comm_rank, comm_size : int
        The local MPI rank and total number of ranks.

    Returns
    -------
    v_tot : double
        ???
    nlev : int
        ???

    """   
    cdef double rat = float(nq) / (nq - 2)
    cdef int nlev = int(np.rint(np.log10(lbox) / np.log10(rat)))   # Essentially log_rat (lbox)
    cdef double total_volume = 0
    cdef int l,i,j,k
    cdef double rlen, rcub

    for l in range(nlev):
        if l % comm_size != comm_rank: continue
        if l == nlev - 1:
            nq += extra           # "Extra" in outermost layer
        rlen = rat**l             # Length of this cube (in which units?). rat > 1, so cubes get larger for successive levels
        rcub = rlen / float(nq-2)   # Length of a cell in the cube.
        
        num_cells = 6*nq*nq - 12*nq + 8
        total_volume += num_cells * rcub**3

        """
        # Find total volume of cubic shell with current cell number and size
        # have nh + 1 = nq (= n) cells per side, so 6n^2 - 12n + 8 cells.
        for k in range(-nh,nh+2,2):
            for j in range(-nh,nh+2,2):
                for i in range(-nh,nh+2,2):
                    ik = max(abs(i),abs(j),abs(k))
                    if ik == nh:
                        total_volume += (rcub)**3.
        """

    return total_volume, nlev


def dicttest(dict gcube):
    print(f"TEST: gcube side length is {gcube['sidelength']}.")


def ndtest(dict gcube, dict scube, dict parts, int comm_rank):
    cdef ndarray[float64_t, ndim=1, mode="c"] masses = parts['m']
    cdef long npart = masses.shape[0]
    cdef long ii = 0
    for ii in range(npart):
        masses[ii] = 555
    return


def fill_scube_layers_new(dict gcube, dict scube, dict nparts,
                      dict parts, int comm_rank, int comm_size):
    """
    Generate Zone III particles for the layers processed on local rank.
    
    Parameters
    ----------
    total_volume : double
        The total volume of regularly assigned cells (including extra cells
        in outer layer).

    The unit length here is the simulation box side length.

    """
    cdef ndarray[float64_t, ndim=1, mode="c"] masses = parts['m']
    cdef ndarray[float64_t, ndim=2, mode="c"] pos = parts['pos']

    cdef double base_l_inner = scube['base_shell_l_inner']
    cdef int n_scells = scube['n_cells']
    cdef int n_extra = scube['n_extra']
    cdef int n_shells = scube['n_shells']
    cdef double l_ratio = scube['l_ratio']
    cdef double leap_mass = scube['leap_mass']

    cdef int ishell, ix, iy, iz
    cdef double x, y, z, m
    cdef double l_inner, scell_size, x_outer
    #cdef ndarray[float64_t, ndim=1, mode="c"] x_1d

    # Initialize (local) index of current particle to generate
    cdef long p_idx = nparts['zone1_local'] + nparts['zone2_local']

    # Loop over shells processed on this rank (inside out) and add particles
    for ishell in np.arange(comm_rank, n_shells, comm_size):
        #print(f"Shell {ishell}...")
        if ishell == n_shells - 1:
            n_scells += n_extra

        l_inner = base_l_inner * l_ratio**ishell
        scell_size = l_inner / (n_scells - 2)
        x_outer = 0.5*l_inner + 0.5*scell_size
        #x_1d = np.linspace(-x_outer, x_outer, num=n_scells)
        #m = scell_size**3
        #if ishell == n_shells - 1:
        #    m += leap_mass

        for iz in range(n_scells):
            for iy in range(n_scells):
                for ix in range(n_scells):
                    #if iz == 0 or iz == n_scells-1 or iy == 0 or iy == n_scells-1:
                    #    xgrid = x_1d
                    #else:
                    #    xgrid = x_1d[[0, -1]]
                    #for ix, x in enumerate(xgrid):
                    if min(ix, iy, iz) == 0 or max(ix, iy, iz) == n_scells - 1:                   
                        masses[p_idx] = m
                        pos[p_idx, 0] = -x_outer + ix * scell_size
                        pos[p_idx, 1] = -x_outer + iy * scell_size
                        pos[p_idx, 2] = -x_outer + iz * scell_size
                        
                        p_idx += 1
        
    np_target = nparts['tot_local']
    if p_idx != np_target:
        raise ValueError(
            f"Ended scube filling with {p_idx} particles, not {np_target}!")


def fill_scube_layers(dict gcube, dict scube, dict nparts,
                      dict parts, int comm_rank, int comm_size):
    """
    Generate Zone III particles for the layers processed on local rank.
    
    Parameters
    ----------
    total_volume : double
        The total volume of regularly assigned cells (including extra cells
        in outer layer).

    The unit length here is the simulation box side length.

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


def fill_scube_layers_old(dict gcube, dict scube, dict nparts,
                          dict parts, int comm_rank, int comm_size):
    """
    Generate Zone III particles for the layers processed on local rank.
    
    Parameters
    ----------
    total_volume : double
        The total volume of regularly assigned cells (including extra cells
        in outer layer).

    The unit length here is the simulation box side length.

    """        
    #double side, int nq, int comm_rank,
    #    int comm_size, int n_tot_lo, int n_tot_hi, int extra, double total_volume,
    #    ndarray[float64_t, ndim=1, mode="c"] coords_x,
    #    ndarray[float64_t, ndim=1, mode="c"] coords_y,
    #    ndarray[float64_t, ndim=1, mode="c"] coords_z,
    #    ndarray[float64_t, ndim=1, mode="c"] masses):

    cdef ndarray[float64_t, ndim=1, mode="c"] masses = parts['m']
    cdef ndarray[float64_t, ndim=2, mode="c"] pos = parts['pos']
    #cdef long npart_tot = masses.shape[0]

    cdef double base_l_inner = scube['base_shell_l_inner']
    cdef int n_scells = scube['n_cells']
    cdef int n_extra = scube['n_extra']
    cdef int n_shells = scube['n_shells']
    cdef double l_ratio = scube['l_ratio']
    cdef double leap_mass = scube['leap_mass']
    cdef long p_idx = nparts['zone1_local'] + nparts['zone2_local']
    cdef int nh = n_scells - 1 
    cdef int ishell,i,j,k,ik
    cdef double l_inner, scell_size
    cdef double m

    # Loop over shells processed on this rank (inside out) and add particles
    for ishell in range(n_shells):#np.arange(comm_rank, n_shells, comm_size):
        if ishell % comm_size != comm_rank:
            continue
    #for ishell in np.arange(comm_rank, n_shells, comm_size):
        #print(f"Shell {ishell}...")
        if ishell == n_shells - 1:
            n_scells += n_extra
            nh = n_scells -1
        l_inner = base_l_inner * l_ratio**ishell
        scell_size = l_inner / float(n_scells - 2)

        m = scell_size**3
        if ishell == n_shells - 1:
            m += leap_mass

        for k in range(-nh, nh+2, 2):
            for j in range(-nh, nh+2, 2):
                for i in range(-nh, nh+2, 2):
                    ik = max(abs(i),abs(j),abs(k))
                    if ik == nh:                      

                        masses[p_idx] = m
                        pos[p_idx, 0] = 0.5 * scell_size * i
                        pos[p_idx, 1] = 0.5 * scell_size * j
                        pos[p_idx, 2] = 0.5 * scell_size * k
                        
                        p_idx += 1
    
    np_target = nparts['tot_local']
    if p_idx != np_target:
        raise ValueError(
            f"Ended scube filling with {p_idx} particles, not {np_target}!")


def fill_gcells_with_particles(
    ndarray[float64_t, ndim=2] gcell_centres,
    ndarray[float64_t, ndim=2] coords_kernel,
    ndarray[float64_t, ndim=2] coords_parts,
    ndarray[float64_t, ndim=1] mass_parts,
    float mass,
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
    mass : float
        The mass to assign to each particle.
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
    >>> _fill_gcells_with_particles(centres, kernel, coords, masses, 0.5, 1)
    >>> print(coords)
    array([[-10., -10., -10.],
           [-1.1, 2.4, 0.4],
           [-0.8, 2.5, 0.2]])
    >>> print(masses)
    array([-10., 0.5, 0.5])

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
                mass_parts[ind] = mass
            count += 1
