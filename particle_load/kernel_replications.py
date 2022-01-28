"""
Collection of functions to replicate a given kernel.

All functions have the same signature:
kernel : ndarray(float)
	The kernel, already expanded for the additional particles. The original
	particles, in the range [-0.5, 0.5], are the first num_orig entries.
ips : float
    The mean inter-particle spacing of the input kernel. In the (implicitly
    assumed, but not neccessary) situation that the kernel is a cubic grid,
    this is equal to the side length of its fundamental cubic cell.
num_orig : int
	The number of particles in the original kernel.

The functions return None; `kernel` is updated in-place.

"""
import numpy as np
from pdb import set_trace

cube_offsets = np.array((
	(0, 0, 0),
	(1, 0, 0),
	(1, 1, 0),
	(0, 1, 0),
	(0, 0, 1),
	(1, 0, 1),
	(1, 1, 1),
	(0, 1, 1)
), dtype=float)

# ------------------  1x replications ---------------------------------------

def replicate_kernel_bcc(kernel, ips, num_orig, mass_ratio):
	"""Simple 1-fold body-centred-cubic replication."""

	kernel_orig = kernel[: num_orig, ...]
	kernel[num_orig : 2*num_orig, ...] = (
		kernel_orig + cube_offsets[6, :] * 0.5 * ips)

	# Compute the centre of mass of each sub-cube, and shift the kernel
	# to bring this back to (0, 0, 0).
	shift = np.array((0.5, 0.5, 0.5)) * ips * (mass_ratio / (1. + mass_ratio))
	kernel -= shift


# ------------------  3x replications ---------------------------------------

def replicate_kernel_n3_faces(kernel, ips, num_orig, mass_ratio):
	"""3-fold replication on faces."""

	kernel_orig = kernel[: num_orig, ...]
	com = np.array((0., 0., 0.))
	for iicube, icube in enumerate([1, 3, 4]):
		kernel[(iicube+1) * num_orig : (iicube+2) * num_orig, ...] = (
			kernel_orig + cube_offsets[icube, :] * 0.5 * ips)
		com += cube_offsets[iicube, :] * 0.5 * ips * mass_ratio

	shift = com / (1. + 3*mass_ratio)
	kernel -= shift


def replicate_kernel_n3_edges(kernel, ips, num_orig, mass_ratio):
	"""3-fold replication on edges."""

	kernel_orig = kernel[: num_orig, ...]
	com = np.array((0., 0., 0.))
	for iicube, icube in enumerate([2, 5, 7]):
		kernel[(iicube+1) * num_orig : (iicube+2) * num_orig, ...] = (
			kernel_orig + cube_offsets[icube, :] * 0.5 * ips)
		com += cube_offsets[iicube, :] * 0.5 * ips * mass_ratio

	shift = com / (1. + 3*mass_ratio)
	kernel -= shift


# ------------------  4x replications ---------------------------------------

def replicate_kernel_n4_faces(kernel, ips, num_orig, mass_ratio):
	"""4-fold replication with 3 replications on faces."""

	kernel_orig = kernel[: num_orig, ...]
	com = np.array((0., 0., 0.))
	for iicube, icube in enumerate([1, 3, 4, 6]):
		kernel[(iicube+1) * num_orig : (iicube+2) * num_orig, ...] = (
			kernel_orig + cube_offsets[icube, :] * 0.5 * ips)
		com += cube_offsets[iicube, :] * 0.5 * ips * mass_ratio

	shift = com / (1. + 4*mass_ratio)
	kernel -= shift


def replicate_kernel_n4_edges(kernel, ips, num_orig, mass_ratio):
	"""4-fold replication with 3 replications on edges."""

	kernel_orig = kernel[: num_orig, ...]
	com = np.array((0., 0., 0.))
	for iicube, icube in enumerate([2, 5, 6, 7]):
		kernel[(iicube+1) * num_orig : (iicube+2) * num_orig, ...] = (
			kernel_orig + cube_offsets[icube, :] * 0.5 * ips)
		com += cube_offsets[iicube, :] * 0.5 * ips * mass_ratio

	shift = com / (1. + 4*mass_ratio)
	kernel -= shift


def replicate_kernel_subsquare(kernel, ips, num_orig, mass_ratio):
	"""4-fold replication with additional particles in a square."""

	offsets = np.array((
		(0.25, 0.25, 0.5),
		(0.75, 0.25, 0.5),
		(0.75, 0.75, 0.5),
		(0.25, 0.75, 0.5)
	), dtype=float) * ips

	kernel_orig = kernel[: num_orig, ...]
	com = np.array((0., 0., 0.))
	for icube in range(0, 4):
		kernel[(icube+1) * num_orig : (icube+2) * num_orig, ...] = (
			kernel_orig + offsets[icube, :])
		com += cube_offsets[icube, :] * mass_ratio

	shift = com / (1. + 4*mass_ratio)
	kernel -= shift


# ------------------  5x replications ---------------------------------------

def replicate_kernel_octahedron(kernel, ips, num_orig, mass_ratio):
	"""5-fold replication with particles arranged in an octahedron."""
	raise Exception("Octahedron replication not yet implemented!")


# ------------------  6x replications ---------------------------------------

def replicate_kernel_n6(kernel, ips, num_orig, mass_ratio):
	"""6-fold replication, additional particles on edges and faces."""

	kernel_orig = kernel[: num_orig, ...]
	com = np.array((0., 0., 0.))
	for iicube, icube in enumerate([1, 2, 3, 4, 5, 7]):
		kernel[(iicube+1) * num_orig : (iicube+2) * num_orig, ...] = (
			kernel_orig + cube_offsets[icube, :] * 0.5 * ips)
		com += cube_offsets[iicube, :] * 0.5 * ips * mass_ratio

	shift = com / (1. + 6*mass_ratio)
	kernel -= shift


# ------------------  7x replications ---------------------------------------

def replicate_kernel_subcube(kernel, ips, num_orig, mass_ratio):
	"""7-fold replication with particles forming a subcube."""

	kernel_orig = kernel[: num_orig, ...]
	com = np.array((0., 0., 0.))
	for icube in range(1, 8):
		kernel[icube * num_orig : icube+1 * num_orig, ...] = (
			kernel_orig + cube_offsets[icube, :] * 0.5 * ips)
		com += cube_offsets[icube, :] * 0.5 * ips * mass_ratio

	shift = com / (1. + 7*mass_ratio)
	kernel -= shift



def replicate_kernel_diamond(kernel, ips, num_orig, mass_ratio):
	"""7-fold replication with particles forming a diamond structure."""

	offsets = np.array((
		(0, 0, 0),
		(0, 2, 2),
		(2, 0, 2),
		(2, 2, 0),
		(3, 3, 3),
		(3, 1, 1),
		(1, 3, 1),
		(1, 1, 3),
	), dtype=float) / 4 * ips

	kernel_orig = kernel[: num_orig, ...]
	com = np.array((0., 0., 0.))
	for iicube in range(1, 8):
		kernel[iicube * num_orig : (iicube+1) * num_orig, ...] = (
			kernel_orig + offsets[iicube, :])
		com += cube_offsets[iicube, :] * mass_ratio

	shift = com / (1. + 7*mass_ratio)
	kernel -= shift