#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import sys
import ctypes
from numpy cimport ndarray, int32_t, float64_t, float32_t, int64_t
import numpy as np
cimport numpy as np

cdef make_grid(
    int n_cells_x, int n_cells_y, int n_cells_z, int comm_rank, int comm_size,
    int num_cells):
    cdef int max_n_cells = max(n_cells_x, n_cells_y, n_cells_z)

    # ** TODO ** Doing this in Cython via a loop is total overkill.
    # It should **definitely** be possible to do this via straightforward
    #

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

cdef _assign_mask_cells(
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


cdef _find_skin_cells(
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

cdef gen_layered_particles_slab(double slab_width, double boxsize, int nq, int nlev, double dv,
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

cdef _guess_nq(double lbox, int nq, int extra, int comm_rank, int comm_size):
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

cdef _gen_layered_particles(double side, int nq, int comm_rank,
        int comm_size, int n_tot_lo, int n_tot_hi, int extra, double total_volume,
        ndarray[float64_t, ndim=1, mode="c"] coords_x,
        ndarray[float64_t, ndim=1, mode="c"] coords_y,
        ndarray[float64_t, ndim=1, mode="c"] coords_z,
        ndarray[float64_t, ndim=1, mode="c"] masses):
    """
    Generate Zone III particles for the layers processed on local rank.
    
    Parameters
    ----------
    total_volume : double
        The total volume of regularly assigned cells (including extra cells
        in outer layer).

    The unit length here is the gcube side length.

    """
    
    cdef double lbox = 1./side         # Simulation box size
    cdef double rat
    cdef int nlev
    cdef long count = 0
    cdef int nh = nq - 1 
    cdef int l,i,j,k,ik,idx
    cdef double rlen, rcub

    # Cube length of one layer in units of the previous layer's cube length.
    # This factor is set by geometry, because shells grow by adding across the
    # old cell's walls.
    rat = float(nq)/(nq-2)

    # Number of layers that comes close to having the box edge exactly on the
    # outer edge of the outermost layer.
    nlev = int(np.log10(lbox)/np.log10(rat)+0.5)
    
    # Difference in volume to make up the mass; this is the volume added to each particle in the outermost layer.
    # Unclear what divisor is meant to represent, should be -4 instead of +2, I think.
    # [2n^2 + 4*((n-2)(n-1))]
    cdef double dv = (lbox**3. - 1**3. - total_volume) /\
            ((nq-1+extra)**2 * 6 + 2)

    if comm_rank == 0:
        print('Rescaling box to %.4f Mpc/h with nq of %i, extra of %i, over %i levels.'\
                %(lbox, nq, extra, nlev))

    # Loop over each level/skin layer.
    for l in range(nlev):
        if l % comm_size != comm_rank: continue
        if l == nlev - 1:
            nq += extra
            nh = nq -1
        rlen = rat**l           # INNER length of this cube.
        rcub = rlen/float(nq-2)   # Length of a cell in the cube. -2 is because of the two 'overhanging' cells at the walls.
        for k in range(-nh,nh+2,2):
            for j in range(-nh,nh+2,2):
                for i in range(-nh,nh+2,2):
                    ik = max(abs(i),abs(j),abs(k))
                    if ik == nh:
                        idx = n_tot_hi + count
                        if l == nlev-1:
                            masses[idx] = rcub**3.+dv
                        else:
                            masses[idx] = rcub**3.
                        coords_x[idx] = 0.5*rcub*i
                        coords_y[idx] = 0.5*rcub*j
                        coords_z[idx] = 0.5*rcub*k
                        count += 1
    
    assert count == n_tot_lo, 'Out particles dont add up %i, %i'%(count, n_tot_lo)
    coords_x[n_tot_hi:] /= lbox
    coords_y[n_tot_hi:] /= lbox
    coords_z[n_tot_hi:] /= lbox
    masses[n_tot_hi:] /= lbox**3.


cdef _fill_gcells_with_particles(
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
    cdef Py_ssize_t ii, jj, kk

    for ii in range(n_gcells):
        for jj in range(np_kernel):
            ind = particle_offset + count
            for kk in range(3):
                coords_parts[ind, kk] = (gcell_centres[ii, kk] +
		                         coords_kernel[jj, kk])
                mass_parts[ind] = mass
            count += 1


# Aliases to interface these functions with the main code

def find_skin_cells(cell_types, cell_nos, L, cell_type):
    return _find_skin_cells(cell_types, cell_nos, L, cell_type)

def guess_nq(lbox, nq, extra, comm_rank, comm_size):
    return _guess_nq(lbox, nq, extra, comm_rank, comm_size)

def assign_mask_cells(mask_pos, mask_cell_width, gcell_pos, gcell_types):
    return _assign_mask_cells(
        mask_pos, mask_cell_width, gcell_pos, gcell_types)

def get_layered_particles_slab(slab_width, boxsize, nq, nlev, dv, comm_rank, comm_size,
        n_tot_lo, n_tot_hi, coords_x, coords_y, coords_z, masses, nq_reduce, extra):
    _gen_layered_particles_slab(slab_width, boxsize, nq, nlev, dv, comm_rank, comm_size,
            n_tot_lo, n_tot_hi, coords_x, coords_y, coords_z, masses, nq_reduce, extra)

def layered_particles(side, nq, comm_rank, comm_size, n_tot_lo, n_tot_hi,
        extra, total_volume, coords_x, coords_y, coords_z, masses):
    _layered_particles(side, nq, comm_rank, comm_size, n_tot_lo, n_tot_hi, extra,
            total_volume, coords_x, coords_y, coords_z, masses)

def fill_gcells_with_particles(
    gcell_centres, coords_kernel, coords_parts, mass_parts, mass, offset):
    return _fill_gcells_with_particles(
        gcell_centres, coords_kernel, coords_parts, mass_parts, mass, offset)


def get_grid(n_cells_x, n_cells_y, n_cells_z, comm_rank, comm_size, num_cells):
    """
    Function to generate a grid...

    Parameters
    ----------
    n_cells_x, n_cells_y, n_cells_z : float
        Number of cells in the x, y, and z dimensions, respectively
    comm_rank : int
        MPI rank of this process.
    comm_size : int
        Total number of MPI ranks.
    num_cells : int
        The number of cells to be processed by this MPI rank.

    Returns
    -------
    offsets : ???
    cell_nos : ???

    """
    return make_grid(
        n_cells_x, n_cells_y, n_cells_z, comm_rank, comm_size, num_cells)
