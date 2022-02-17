"""Routines for matching indices between two data sets."""

import numpy as np
import os
import time
import ctypes as c

from pdb import set_trace


class Gate:
    """Class for linking two catalogues through a common set of keys.

    It provides functionality to translate indices in one ('external',
    subject) catalogue into indices in a second ('internal', reference)
    catalogue, for key sets of arbitrary length or values.

    Parameters
    ----------
    ids_ext : ndarray(int)
        First, 'external' set of IDs (keys).

    ids_int : ndarray(int) or ``None``, optional
        Second, 'internal' set of IDs (keys). If it is not provided
        (``None``, default), the corresponding reverse-ID list must be
        (`rev_IDs`).
    rev_IDs : ndarray(int), `ReverseList`, or ``None``, optional
        The reverse-ID list for the second, internal data set. It can
        either be supplied as a plain (int) array or as a `ReverseList`
        object. If it is not provided (``None``, default), the internal IDs
        must be supplied as `ids_int`.

    max_direct : int, optional
        Maximum *value* in either of the key sets for which a reverse-index
        based match can be performed (default: 1e10). For larger values,
        keys are always matched via sorting. This parameter has no
        effect if a reverse ID list is provided as `rev_IDs`.
    sort_below : int, optional
        Maximum *length* of either key set for which the sort-based search
        is always done (default: 100). Ignored if `rev_IDs` is provided.
    force_sort : bool, optional
        Force a sort-based search irrespective of input (default:
        ``False``). This is equivalent to, but slightly faster than,
        setting `max_direct` = 0. Ignored if `rev_IDs` is provided.
    min_c : int, optional
        Minimum length of (longer) key set for which a (possible)
        sort-based search is outsourced to the C-library (default: 1e5),
        if available. This parameter has no effect if a reverse ID list is
        provided as `rev_IDs`.
    save_revIDs : bool, optional
        Internally store the reverse ID list, if it was provided or
        created internally (default: ``False``). This can be useful if one
        data set must be matched to multiple others, in which case the
        reverse ID list can be re-used.
    verbose : bool, optional
        Provide diagnostic messages (default: ``False``)

    Attributes
    ----------
    rev_IDs : ReverseList instance or ``None``
        The reversed internal ID (key) list (``None`` if not created or
        not stored).
    """

    def __init__(self, ids_ext, ids_int=None, rev_IDs=None,
                 max_direct=int(1e10), sort_below=100, force_sort=False,
                 min_c=int(1e5), save_revIDs=False, verbose=False):
        if verbose:
            print("Beginning gate setup...")

        self.rev_IDs = None   # Default for consistency

        if rev_IDs is None:
            # Need to set up matching instructions from scratch
            if ids_int is None:
                print("Cannot set up a gate without any info on internal "
                      "data set...")
                set_trace()

            self.num_int = len(ids_int)

            # Can we use the direct (reverseID-list-based) method?
            use_direct = True
            if force_sort:
                use_direct = False
            elif sort_below is not None:
                if max(len(ids_ext), len(ids_int)) < sort_below:
                    use_direct = False
            elif max(np.max(ids_ext), np.max(ids_int)) > max_direct:
                use_direct = False

            if use_direct:
                # Quick and simple way via reverse ID list.
                rev_IDs = ReverseList(ids_int)
                self.ie_int, self.ie_matched = rev_IDs.query_matched(ids_ext)
                if save_revIDs:
                    self.rev_IDs = rev_IDs

            else:
                # Sort-based match must be done on a case-by-case basis,
                # so can directly use find_id_indices()
                self.ie_int, self.ie_matched = find_id_indices(
                    ids_ext, ids_int, force_sort=True, verbose=verbose,
                    min_c=min_c)

        else:
            # Simplest case: we were provided with a reverse list
            if save_revIDs:
                self.rev_IDs = rev_IDs

            # Account for possibility of revID array being too short
            self.ie_int = query_array(rev_IDs, ids_ext)
            self.ie_matched = np.nonzero(self.ie_int >= 0)[0]

            self.num_int = np.max(rev_IDs) + 1

    def in_int(self, index=None):
        """
        Return the indices of the external keys in the internal catalogue.

        Parameters
        ----------
        index : ndarray(int), optional
            Subset of the external catalogue for which the indices should
            be returned. If ``None`` (default), return indices for the
            entire external catalogue.

        Returns
        -------
        ie_int : ndarray(int)
            The indices into the internal catalogue corresponding to the
            queried external elements (-1 for those that are not matched).
        ie_matched : ndarray(int)
            The indices into the queried external elements that could be
            matched to an internal key.
        """
        if index is None:
            return self.ie_int, self.ie_matched

        ie_matched_index = np.nonzero(self.ie_int[index] >= 0)[0]
        return self.ie_int[index], ie_matched_index

    def in_ext(self, index=None):
        """
        Return the indices of the internal keys in the external catalogue.

        Parameters
        ----------
        index : ndarray(int), optional
            Subset of the internal catalogue for which the indices should
            be returned. If ``None`` (default), return indices for the
            entire internal catalogue.

        Returns
        -------
        ii_ext : ndarray(int)
            The indices into the external catalogue corresponding to the
            queried internal elements (-1 for those that are not matched).
        ii_matched : ndarray(int)
            The indices into the queried internal elements that could be
            matched to an external key.
        """
        ii_ext = np.zeros(self.num_int, dtype=int)-1
        ii_matched = self.ie_int[self.ie_matched]
        ii_ext[ii_matched] = self.ie_matched

        if index is None:
            return ii_ext, ii_matched

        ii_matched_index = np.nonzero(ii_ext[index] >= 0)[0]
        return ii_ext[index], ii_matched_index


class ReverseList:
    """Class for creating and querying a reverse-index list.

    This is essentially a thin wrapper around the `create_reverse_list()`
    function, but avoids the need to artificially expand the list
    to deal with possible out-of-bounds queries. It creates and queries
    an array containing the index for each ID value, i.e. the inverse of
    the input list (which gives the ID for each index).

    Warning
    -------
    It is a bad idea to instantiate this class with a key set that contains
    large values in relation to the available memory: as a rough guide,
    consider the limit as 1/8 * [RAM/byte]. Ignoring this will result in
    undefined, slow, and likely annoying behaviour.

    Parameters
    ----------
    ids : ndarray (int)
        Input keys (IDs) to invert. These are assumed to be unique;
        death and destruction may happen if this is not the case.
        The array must be one-dimensional. Any negative values are
        assumed to be dummy elements and are ignored.
    delete_ids : bool, optional
        Delete input keys after building reverse list (default: ``False``).
    assume_positive : bool, optional
        Assume that all the input values are non-negative, which speeds
        up the inversion. If ``False`` (default), the code checks
        explicitly which input values are non-negative and only
        transcribes these to the reverse list.
    compact : bool, optional
        Make the reverse-index list shorter by (transparently) subtracting
        the minimum ID from inputs (default: ``False``).

    Attributes
    ----------
    reverse_IDs : ndarray(int)
        The reverse-index array, created upon instantiation.
    num_int : int
        The number of keys in the input array.
    """

    def __init__(self, ids, delete_ids=False, assume_positive=False,
                 compact=False):
        self.num_int = len(ids)

        # Find and subtract minimum (positive) value from input keys
        if compact:
            if assume_positive:
                self.min_key = np.min(ids)
            else:
                ind_positive = np.nonzero(ids >= 0)[0]
                self.min_key = np.min(ids[ind_positive])
            self.reverse_IDs = create_reverse_list(
                ids - self.min_key, delete_ids=delete_ids,
                assume_positive=assume_positive)
        else:
            # Invert unmodified keys
            self.min_key = None
            self.reverse_IDs = create_reverse_list(
                ids, delete_ids=delete_ids, assume_positive=assume_positive)

    def query(self, ids, assume_valid=False):
        """Find the indices of the input (external) IDs.

        Parameters
        ----------
        ids : ndarray(int)
            The IDs whose indices in the internal list (used to set up the
            reverse list) should be determined. Out-of-bound situations are
            dealt with internally.
        assume_valid : bool, optional
            Assume that all input IDs are within the range of the internal
            reverse-index list, so that out-of-bound check can be skipped
            (default: ``False``).

        Returns
        -------
        indices : ndarray(int)
            The indices corresponding to each input ID (-1 if not found).

        Note
        ----
        This is also the object's call method, so it can be used
        directly without specifying `query`.

        Example
        -------
        >>> import numpy as np
        >>> ids = np.array([4, 0, 5, 1])
        >>> reverse_ids = ReverseList(ids)
        >>> reverse_ids(np.array([3, 4]))
        array([-1, 0])
        """
        if assume_valid:
            if self.min_key is None:
                return self.reverse_IDs[ids]
            else:
                return self.reverse_IDs[ids - self.min_key]
        else:
            if self.min_key is None:
                return query_array(self.reverse_IDs, ids)
            else:
                return query_array(self.reverse_IDs, ids - self.min_key)

    def __call__(self, ids, assume_valid=False):
        """Alias for query function."""
        return self.query(ids, assume_valid=assume_valid)

    def query_matched(self, ids, assume_valid=False):
        """Find indices eof the input (external) IDs, also listing matches.

        Parameters
        ----------
        ids : ndarray(int)
            The IDs whose indices in the internal list (used to set up the
            reverse list) should be determined. Out-of-bound situations are
            dealt with internally.
        assume_valid : bool, optional
            Assume that all input IDs are within the range of the internal
            reverse-index list, so that out-of-bound check can be skipped
            (default: ``False``).

        Returns
        -------
        indices : ndarray(int)
            The indices corresponding to each input ID (-1 if not found).
        matches : ndarray(int)
            The indices into the input `ids` array for keys that could be
            matched (i.e. have a non-negative value of `indices`).

        Example
        -------
        >>> import numpy as np
        >>> ids = np.array([4, 0, 5, 1])
        >>> reverse_ids = ReverseList(ids)
        >>> reverse_ids.query_matched(np.array([3, 4]))
        array([-1, 0]), array([1])
        """
        ie_int = self.query(ids, assume_valid=assume_valid)
        ie_matched = np.nonzero(ie_int >= 0)[0]
        return ie_int, ie_matched


def create_reverse_list(ids, delete_ids=False, assume_positive=False,
                        max_val=None):
    """Create a reverse-index list from a (unique) list of IDs.

    This gives the index for each ID value, and is thus the inverse of the
    input list (which gives the ID for an index).

    Warning
    -------
    It is a bad idea to call this function if the input ID list contains very
    large values in relation to the available memory: as a rough guide,
    consider the limit as 1/8 * [RAM/byte]. Ignoring this will result in
    undefined, slow, and likely annoying behaviour.

    Parameters
    ----------
    ids : ndarray (int)
        Input keys (IDs) to invert. These are assumed to be unique;
        death and destruction may happen if this is not the case.
        The array must be one-dimensional. Any negative values are
        assumed to be dummy elements and are ignored.
    delete_ids : bool, optional
        Delete input keys after building reverse list (default: ``False``).
    assume_positive : bool, optional
        Assume that all the input values are non-negative, which speeds
        up the inversion. If ``False`` (default), the code checks
        explicitly which input values are non-negative and only
        transcribes these to the reverse list.
    max_val : int, optional
        Build a reverse list with at least `maxval`+1 entries (i.e. that
        can be indexed by values up to `maxval`), even if this exceeds the
        maximum input ID value. If ``None`` (default), the reverse list is
        built up to self-indexing capacity (i.e. with max(`ids`)+1
        elements).

    Returns
    -------
    rev_IDs : ndarray(int)
        The reverse index list. If the input list contains fewer than
        2 billion elements, it is of type np.int32, otherwise np.int64.

    Note
    ----
    For most practical purposes, it may be more convenient to directly use
    the Gate or ReverseList classes, or the find_id_indices() function,
    to correlate IDs, all of which call this function internally.

    Example
    -------
    >>> # Standard use to invert an array:
    >>> import numpy as np
    >>> ids = np.array([4, 0, 5, 1])
    >>> create_reverse_list(ids)
    array([1, 3, -1, -1, 0, 2])

    >>> # Including negative array values:
    >>> ids = np.array([4, 0, -1, 5, 1])
    >>> create_reverse_list(ids)
    array([1, 4, -1, -1, 0, 3])

    >>> # Including use of max_val:
    >>> create_reverse_list(ids, maxVal=10)
    array([1, 4, -1, -1, 0, 3, -1, -1, -1, -1, -1])
    """
    # If there is no input, return an empty array
    if len(ids) == 0:
        if max_val is None:
            return np.zeros(0, dtype=np.int32)
        else:
            return np.zeros(max_val+1, dtype=np.int32) - 1

    # Find extent and required data type of output list
    max_ID = ids.max()
    if max_val is not None:
        if max_val > max_ID:
            max_ID = max_val
    if len(ids) > 2e9:
        dtype = np.int64
    else:
        dtype = np.int32

    # Do the actual inversion
    rev_IDs = np.zeros(np.int64(max_ID+1), dtype=dtype) - 1
    if assume_positive:
        rev_IDs[ids] = np.arange(len(ids))
    else:
        ind_good = np.nonzero(ids >= 0)[0]
        rev_IDs[ids[ind_good]] = ind_good

    # Delete input if necessary, to save space
    if delete_ids:
        del ids

    return rev_IDs


def find_id_indices(ids_ext, ids_int, max_direct=int(1e10), min_c=int(1e5),
                    sort_below=None, force_sort=False, sort_matches=True,
                    verbose=False):
    """Find and return the locations of IDs in a reference list.

    This function can be used to translate indices in one ('external',
    subject) catalogue into indices in a second ('internal', reference)
    catalogue, for key sets of arbitrary length or values. If the maximum
    value is below an (adjustable) threshold, the lookup is performed
    via an explicit reverse-lookup list. Otherwise, both input and reference
    lists are sorted and compared via a "Katamaran-search".

    Parameters
    ----------
    ids_ext : ndarray (int)
        Array of keys (IDs) whose indices in `ids_int` should be returned.
        It should be unique, unless the search is guaranteed to be
        executed via a reverse list.
    ids_int : ndarray (int)
        Internal ('reference') list of keys (IDs), assumed to be unique.
        The function will search for each input ID in this array.
    max_direct : int, optional
        Maximum value in either key set for which a reverse-index
        based match is performed, rather than a sort-based search
        (default: 1e10).
    min_c : int, optional
        Minimum length of longer ID list for which a (possible)
        Katamaran-search is outsourced to the C-library, if compiled
        (default: 1e5). Ignored if a direct search is performed.
    sort_below : int or ``None``, optional
        Maximum length of either ID list for which sort-based search is
        preferred (default: 100). If ``None``, always use reverse-index
        method if possible.
    force_sort : bool, optional
        Force a sorted search irrespective of maximum input values. This
        is equivalent to, but slightly faster than, setting
        `max_direct` = 0.
    sort_matches : bool, optional
        Explicitly sort matching indices from Katamaran-search in ascending
        order, so that its result is identical to reverse-list based
        method (default: ``True``).
    verbose : bool, optional
        Print timing information (default: ``False``)

    Returns
    -------
    ie_int : ndarray(int)
        The index in ids_int for each input ID (-1 if it could not be
        located at all).
    ie_matched : ndarray(int)
        The input (external) ID indices that could be matched.

    Note
    ----
    For large arrays, using a reverse-lookup list is typically much faster,
    but may use substantial amounts of memory. For small input lists, the
    sort-and-search approach may be faster because it avoids the overheads
    of setting up the reverse list.
    """
    # Determine whether we can use the direct (reverse-ID-list-based) method
    use_direct = True
    if force_sort:
        use_direct = False
    elif sort_below is not None:
        if max(len(ids_ext), len(ids_int)) > sort_below:
            use_direct = False
    elif max(np.max(ids_ext), np.max(ids_int)) > max_direct:
        use_direct = False

    if use_direct:
        # Quick and simple way via reverse ID list:
        rev_IDs = ReverseList(ids_int)
        ie_int = rev_IDs.query(ids_ext)
        ie_matched = np.nonzero(ie_int >= 0)[0]

    else:
        # Need to identify matching IDs on an individual basis.
        time_start = time.time()
        sorter_ext = np.argsort(ids_ext)
        sorter_int = np.argsort(ids_int)
        if verbose:
            time_sort = time.time() - time_start
            time_start = time.time()

        if ((max(len(ids_ext), len(ids_int)) > min_c) and
            os.path.isfile(os.path.dirname(os.path.realpath(__file__)) +
                           "/clib/ckat.so")):
            # Long input list(s): use C version of Katamaran
            ise_sorted_int = cKatamaran_search(ids_ext[sorter_ext],
                                               ids_int[sorter_int])
        else:
            # Short input lists: use python version of Katamaran
            ise_sorted_int = katamaran_search(ids_ext[sorter_ext],
                                              ids_int[sorter_int])

        # Convert to matches of (unsorted) ext in (unsorted) int
        ise_matched = np.nonzero(ise_sorted_int >= 0)[0]
        ie_matched = sorter_ext[ise_matched]
        ie_int = np.zeros(len(ids_ext), dtype=int) - 1
        ie_int[ie_matched] = sorter_int[ise_sorted_int[ise_matched]]

        # If desired, sort matches (only at the end, to not break
        # alignment between ie_matched and ise_matched!)
        if sort_matches:
            ie_matched = np.sort(ie_matched)

        if verbose:
            time_kat = time.time() - time_start
            print("Sorting took    {:.3f} sec." .format(time_sort))
            print("Kat-search took {:.3f} sec." .format(time_kat))

    return ie_int, ie_matched


def query_array(array, indices, default=-1):
    """Retrieve values from an array that may be too short.

    Parameters
    ----------
    array : ndarray
        The array to retrieve values from. It should be of numerical
        type, and one-dimensional.
    indices : ndarray(int)
        Indices of array to query.
    default : value, optional
        The value to assign to out-of-bound indices (default: -1).

    Returns
    -------
    values : ndarray
        The array values of the specified indices, with out-of-bounds
        indices (negative or >= len(`array`)) set to `default`.

    Example
    -------
    >>> import numpy as np
    >>> arr = np.array([1, 4, 0, 8])
    >>> ind = np.array([0, 2, 8])
    >>> query_array(arr, ind, default=-100)
    array([1, 0, -100])
    """
    if np.min(indices) >= 0 and np.max(indices) < len(array):
        return array[indices]
    else:
        values = np.zeros(len(indices), dtype=array.dtype) + default
        ind_in_range = np.nonzero((indices < len(array)) &
                                  (indices >= 0))[0]
        values[ind_in_range] = array[indices[ind_in_range]]
        return values


def katamaran_search(a, b):
    """Perform a "Katamaran" search to locate elements of a in b.

    This assumes that the elements in a and b are unique, and
    that a and b are sorted.

    Parameters
    ----------
    a : ndarray
        Elements to locate. Must be unique and sorted.
    b : ndarray
        Reference values. Must also be unique and sorted.

    Returns
    -------
    ind_a : ndarray(int)
        Indices of a in b (-1 for elements that could not be matched).

    Note
    ----
    a and b should be of the same data type. If they contain floats,
    the matching values must be exactly the same to machine precision.

    This function is implemented in pure python and may be very slow for
    large arrays. Consider using `cKatamaran_search()` instead for these.
    """
    if len(a) == 0:
        return np.zeros(0, dtype=int)-1

    locs_a = np.zeros(len(a), dtype=int)-1

    if len(b) == 0:
        return locs_a

    # Initialize search indices and values
    ind_a = ind_b = 0
    len_a, len_b = len(a), len(b)
    val_a, val_b = a[ind_a], b[ind_b]

    # Perform main search as infinite loop, break when done
    while True:
        # a ahead of b: test next b element
        if val_a > val_b:
            ind_b += 1
            if ind_b >= len_b:
                break
            val_b = b[ind_b]
            continue

        # b ahead of a: test next a element
        if val_b > val_a:
            ind_a += 1
            if ind_a >= len_a:
                break
            val_a = a[ind_a]
            continue

        # Match: record and increase both
        if val_a == val_b:
            locs_a[ind_a] = ind_b
            ind_a += 1
            ind_b += 1
            if ind_a >= len_a or ind_b >= len_b:
                break
            val_a = a[ind_a]
            val_b = b[ind_b]
            continue

    return locs_a


def cKatamaran_search(a, b):
    """Sped-up "Katamaran-search", which uses an external C library.

    This assumes that the elements in a and b are unique, and
    that a and b are sorted.

    Parameters
    ----------
    a : ndarray
        Elements to locate. Must be unique and sorted.
    b : ndarray
        Reference values. Must also be unique and sorted.

    Returns
    -------
    ind_a : ndarray(int)
        Indices of a in b (-1 for elements that could not be matched).

    Note
    ----
    a and b should be of the same data type. If they contain floats,
    the matching values must be exactly the same to machine precision.

    For small arrays, the overhead of interfacing to the C code may
    negate the speed benefit; consider using `katamaran_search()` here.
    """
    if len(a) == 0:
        return np.zeros(0, dtype=np.int32)-1

    locs_a = np.zeros(len(a), dtype=np.int64)-1

    if len(b) == 0:
        return locs_a

    # Set up C library
    object_dir = os.path.dirname(os.path.realpath(__file__)) + "/clib/"
    lib = c.cdll.LoadLibrary(object_dir + 'ckat.so')
    ckat = lib.ckat

    # Convert variables to C pointers
    a_for_c = a.astype(np.int64)
    b_for_c = b.astype(np.int64)
    a_p = a_for_c.ctypes.data_as(c.c_void_p)
    b_p = b_for_c.ctypes.data_as(c.c_void_p)
    locs_a_p = locs_a.ctypes.data_as(c.c_void_p)
    na_c = c.c_long(len(a))
    nb_c = c.c_long(len(b))

    # Call C routine to get results
    myargv = c.c_void_p * 5
    argv = myargv(a_p, b_p, c.addressof(na_c), c.addressof(nb_c), locs_a_p)
    ckat(5, argv)

    return locs_a
