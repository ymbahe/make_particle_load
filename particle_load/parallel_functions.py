import numpy as np


def my_alltoallv(sendbuf, send_count, send_offset, recvbuf, recv_count, recv_offset, comm,
        comm_rank, comm_size):
    """Alltooallv implemented using sendrecv calls"""

    # Get communicator to use
    ptask = 0
    while(2**ptask < comm_size):
        ptask += 1

    # Loop over pairs of processes and send data
    for ngrp in range(2**ptask):
        rank = comm_rank ^ ngrp
        if rank < comm_size:
            soff = send_offset[rank]
            sc   = send_count[rank]
            roff = recv_offset[rank]
            rc   = recv_count[rank]
            if sc > 0 or rc > 0:
                comm.Sendrecv(sendbuf[soff:soff+sc], rank, 0,
                              recvbuf[roff:roff+rc], rank, 0)

def repartition(arr, ndesired, comm, comm_rank, comm_size):
    """
    Return the input arr repartitioned across MPI ranks.

    Parameters
    ----------
    arr : ndarray
        The input array; must be 1-dimensional.
    ndesired : ndarray(int)
        The target number of elements on each rank.

    Returns
    -------
    arr_new : ndarray
        The repartitioned array, with ndesired[i+1] - ndesired[i] number
        of elements on rank i.

    """

    # Make sure input is a contiguous array
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)

    # Sanity checks on input arrays
    if len(arr.shape) != 1:
        print("Can only repartition 1D array data! (shape=%i)"%len(arr.shape))
        comm.Abort()

    # Find total number of elements
    n        = arr.shape[0]
    nperproc = comm.allgather(n)
    ntot     = sum(nperproc)

    # Check total doesn't change
    if ntot != sum(ndesired):
        print("repartition() - number of elements must be conserved!")
        comm.Abort()

    # Find first index on each processor
    first_on_proc_in  = np.cumsum(nperproc) - nperproc
    first_on_proc_out = np.cumsum(ndesired) - ndesired

    # Count elements to go to each other processor
    send_count = np.zeros(comm_size, dtype=np.int64)
    for rank in range(comm_size):
        # Find range of elements to go to this other processor
        ifirst = first_on_proc_out[rank]
        ilast  = ifirst + ndesired[rank] - 1
        # We can only send the elements which are stored locally
        ifirst = max((ifirst, first_on_proc_in[comm_rank]))
        ilast  = min((ilast,  first_on_proc_in[comm_rank] +
                      nperproc[comm_rank] - 1))
        send_count[rank] = np.max((0,ilast-ifirst+1))

    # Transfer the data
    send_displ = np.cumsum(send_count) - send_count
    recv_count = np.ndarray(comm_size, dtype=np.int64)
    comm.Alltoall(send_count, recv_count)
    recv_displ = np.cumsum(recv_count) - recv_count
    arr_return = np.ndarray(sum(recv_count), dtype=arr.dtype)
    my_alltoallv(arr, send_count, send_displ, arr_return, recv_count, recv_displ, comm,
            comm_rank, comm_size)

    return arr_return
