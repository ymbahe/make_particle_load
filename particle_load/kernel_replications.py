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

# ------------------  1x replications ---------------------------------------

def replicate_kernel_bcc(kernel, ips, num_orig, mass_ratio):
	"""Simple 1-fold body-centred-cubic replication."""

	kernel_orig = kernel[num_orig : 2*num_orig, ...]
	kernel[num_orig : 2*num_orig, ...] = (
		kernel_orig + np.array((0.5, 0.5, 0.5)) * ips)

	# Compute the centre of mass of each sub-cube, and shift the kernel
	# to bring this back to (0, 0, 0).
	shift = np.array((0.5, 0.5, 0.5)) * ips * (mass_ratio / (1. + mass_ratio))
	kernel -= shift


# ------------------  4x replications ---------------------------------------

def replicate_kernel_n4_faces(kernel, ips, num_orig):
	"""4-fold replication with 3 replications on faces."""
	raise Exception("N4-faces replication not yet implemented!") 


def replicate_kernel_n4_edges(kernel, ips, num_orig):
	"""4-fold replication with 3 replications on edges."""
	raise Exception("N4-edges replication not yet implemented!") 


def replicate_kernel_subsquare(kernel, ips, num_orig):
	"""4-fold replication with additional particles in a square."""
	raise Exception("Subsquare replication not yet implemented!")


# ------------------  5x replications ---------------------------------------

def replicate_kernel_octahedron(kernel, ips, num_orig):
	"""5-fold replication with particles arranged in an octahedron."""
	raise Exception("Octahedron replication not yet implemented!")


# ------------------  6x replications ---------------------------------------

def replicate_kernel_n6(kernel, ips, num_orig):
	"""6-fold replication, additional particles on edges and faces."""
	raise Exception("N6 replication not yet implemented!")


# ------------------  7x replications ---------------------------------------

def replicate_kernel_subcube(kernel, ips, num_orig):
	"""7-fold replication with particles forming a subcube."""
	raise Exception("Subcube replication not yet implemented!")


def replicate_kernel_diamond(kernel, ips, num_orig):
	"""7-fold replication with particles forming a diamond structure."""
	raise Exception("Diamond replication not yet implemented!")