import numpy as np
import h5py
import os

class read_swift:

    def __init__(self, fname, comm=None, verbose=False, distributed_format='collective',
            max_concur_io = 64):

        # Talkative?
        self.verbose = verbose

        # When snapshots are distributed over multiple files.
        # -- 'non_collective'   = Each file part is assigned to a core.
        # -- 'collective'       = Each file part read collectivley using all cores.
        self.distributed_format = distributed_format

        # Has select region or split selection been called?
        self.region_selected_on = -1
        self.split_selection_called = False

        # Snapshot file to load.
        self.fname = fname
        assert os.path.isfile(self.fname), 'File %s does not exist.'%fname

        # Dont load more than this amount at once.
        self.max_size_to_read_at_once = 2 * 1024. * 1024 * 1024 # 2 Gb in bytes

        # Get information from the header.
        self.read_header()

        # MPI communicator
        self.comm = comm
        self.max_concur_io = max_concur_io          # Max number of cores that can read at once.
        if self.comm is None:
            self.comm_rank = 0
            self.comm_size = 1
        else:
            self.comm_rank = self.comm.Get_rank()
            self.comm_size = self.comm.Get_size()

    def message(self, m):
        if self.comm is None:
            print(m)
        else:
            if self.comm.Get_rank() == 0:
                print(m)

    def split_selection(self):
        """ Divides the particles between cores. """

        if self.HEADER['NumFilesPerSnapshot'] > 1:
            if self.distributed_format == 'non_collective':
                self.split_selection_distributed()
            else:
                self.split_selection_collective()
        else:
            self.split_selection_collective()

    def split_selection_distributed(self):
        """ Splits selected particles between cores. Equal number of distributed files per core """

        if self.verbose: self.message("Split selection is distributed.")
        assert self.region_selected_on != -1, 'Need to call select region before split_selection'

        self.index_data = {}
        self.index_data['lefts'] = []
        self.index_data['rights'] = []
        self.index_data['files'] = []
        self.index_data['num_to_load'] = []
        self.index_data['total_num_to_load'] = 0

        for fileno in np.unique(self.region_data['files']):
            if fileno % self.comm_size != self.comm_rank: continue
            mask = np.where(self.region_data['files'] == fileno)

            self.index_data['files'].append(fileno)
            self.index_data['lefts'].append(self.region_data['lefts'][mask])
            self.index_data['rights'].append(self.region_data['rights'][mask])
            self.index_data['num_to_load'].append(self.region_data['num_to_load'][mask])
            self.index_data['total_num_to_load'] += np.sum(self.index_data['num_to_load'][-1])

        self.split_selection_called = True

    def split_selection_collective(self):
        """ Splits selected particles between cores. Equal number of particles per core """

        if self.verbose: self.message("Split selection is collective.")
        assert self.region_selected_on != -1, 'Need to call select region before split_selection'

        self.index_data = {}
        self.index_data['lefts'] = []
        self.index_data['rights'] = []
        self.index_data['files'] = []
        self.index_data['num_to_load'] = []
        self.index_data['total_num_to_load'] = 0

        if self.comm_size > 1 and self.region_data['total_num_to_load'] > 100*self.comm_size:

            # Loop over each file.
            for fileno in np.unique(self.region_data['files']):
                mask = np.where(self.region_data['files'] == fileno)

                # How many particles from this file will each core load?
                num_to_load_from_this_file = np.sum(self.region_data['num_to_load'][mask])
                num_per_core = num_to_load_from_this_file // self.comm_size
                my_num_to_load = num_per_core
                if self.comm_rank == self.comm_size-1:
                    my_num_to_load += num_to_load_from_this_file % self.comm_size

                # The case where this file has few particles compared to the number of ranks.
                if num_per_core < 100:
                    if self.comm_rank == self.comm_size - 1: 
                        if self.verbose:
                            print('Rank %i will load %i particles l=%s r=%s from file=%i'\
                                    %(self.comm_rank, my_num_to_load,
                                        tmp_my_lefts, tmp_my_rights, fileno))
        
                        self.index_data['lefts'].append([0])
                        self.index_data['rights'].append([my_num_to_load])
                        self.index_data['files'].append(fileno)
                        self.index_data['num_to_load'].append([my_num_to_load])
                        self.index_data['total_num_to_load'] += my_num_to_load

                # The case where this file has many particles per rank.
                else:
                    # What particles will each core load?
                    tmp_my_lefts = []
                    tmp_my_rights = []
                    tmp_my_files = []
                    count = 0       # How many total particles have been loaded.
                    my_count = np.zeros(self.comm_size, dtype='i8') # How many ps have I loaded?

                    for i,(l,r,chunk_no) in enumerate(zip(self.region_data['lefts'][mask],
                                                       self.region_data['rights'][mask],
                                                       self.region_data['num_to_load'][mask])):
                        mask = np.where(my_count < num_per_core)
                        
                        # How many cores will this chunk be spread over?
                        num_cores_this_chunk = \
                            ((chunk_no + my_count[np.min(mask)]) // num_per_core) + 1
                        chunk_bucket = chunk_no
                        for j in range(num_cores_this_chunk):
                            if np.min(mask)+j < self.comm_size:
                                if my_count[np.min(mask)+j] + chunk_bucket > num_per_core:
                                    if self.comm_rank == np.min(mask)+j:
                                        tmp_my_lefts.append(l + chunk_no - chunk_bucket)
                                        tmp_my_rights.append(l + chunk_no - chunk_bucket + \
                                                num_per_core - my_count[np.min(mask)+j])
                                        tmp_my_files.append(fileno)
                                    chunk_bucket -= (num_per_core - my_count[np.min(mask)+j])
                                    my_count[np.min(mask)+j] = num_per_core
                                else:
                                    if self.comm_rank == np.min(mask)+j:
                                        tmp_my_lefts.append(l + chunk_no - chunk_bucket)
                                        tmp_my_rights.append(r)
                                        tmp_my_files.append(fileno)
                                    my_count[np.min(mask)+j] += chunk_bucket
                            else:
                                if self.comm_rank == self.comm_size - 1:
                                    if my_count[-1] < my_num_to_load:
                                        tmp_my_rights[-1] += chunk_bucket
                                        my_count[-1] += chunk_bucket
                  
                    # Make sure we got them all.
                    assert self.comm.allreduce(
                        np.sum(np.array(tmp_my_rights)-np.array(tmp_my_lefts))) \
                       == num_to_load_from_this_file, 'Did not divide up the particles correctly'
        
                    if self.verbose:
                        self.comm.barrier()
                        print('Rank %i will load %i particles l=%s r=%s from file=%i'\
                                %(self.comm_rank, my_num_to_load,
                                    tmp_my_lefts, tmp_my_rights, fileno))
    
                    self.index_data['lefts'].append(tmp_my_lefts)
                    self.index_data['rights'].append(tmp_my_rights)
                    self.index_data['files'].append(fileno)
                    self.index_data['num_to_load'].append(np.array(
                        self.index_data['rights'][-1])-np.array(self.index_data['lefts'][-1]))
                    self.index_data['total_num_to_load'] += \
                            np.sum(self.index_data['num_to_load'][-1])
        else:
            # Single core case.
            for fileno in np.unique(self.region_data['files']):
                mask = np.where(self.region_data['files'] == fileno)

                self.index_data['files'].append(fileno)
                self.index_data['lefts'].append(self.region_data['lefts'][mask])
                self.index_data['rights'].append(self.region_data['rights'][mask])
                self.index_data['num_to_load'].append(self.region_data['num_to_load'][mask])
                self.index_data['total_num_to_load'] += np.sum(self.index_data['num_to_load'][-1])

        self.split_selection_called = True

    def find_select_region_cells(self, f, x_min, x_max, y_min, y_max, z_min, z_max, eps=1e-4):
        """ See what TL cells our region intersects. """

        centres = f["/Cells/Centres"][...]
        size = f["/Cells/Meta-data"].attrs["size"]
        
        # Coordinates to load around.
        coords = np.array([ x_min + (x_max - x_min)/2.,
                            y_min + (y_max - y_min)/2.,
                            z_min + (z_max - z_min)/2.])
      
        # Wrap to given coordinates.
        boxsize = self.HEADER['BoxSize']
        centres = np.mod(centres-coords+0.5*boxsize, boxsize)+coords-0.5*boxsize

        # Find what cells fall within boundary.
        dx_over_2 = (x_max - x_min)/2. + eps
        dy_over_2 = (y_max - y_min)/2. + eps    
        dz_over_2 = (z_max - z_min)/2. + eps
        half_size = size / 2.       # Half a cell size.

        mask = np.where(
                (centres[:,0] + half_size[0] >= coords[0] - dx_over_2) &
                (centres[:,0] - half_size[0] <= coords[0] + dx_over_2) &
                (centres[:,1] + half_size[1] >= coords[1] - dy_over_2) &
                (centres[:,1] - half_size[1] <= coords[1] + dy_over_2) &
                (centres[:,2] + half_size[2] >= coords[2] - dz_over_2) &
                (centres[:,2] - half_size[2] <= coords[2] + dz_over_2))

        return mask

    def select_region(self, part_type, x_min, x_max, y_min, y_max, z_min, z_max,
                      just_load_all=False):
        """ Select what cells contrain the particles in a cube around passed coordinates. """

        # Look at the passed file, work out the structure.
        mask = None
        just_load_all = None

        if self.comm_rank == 0:

            if self.verbose:
                print('Selection region x=%.4f->%.4f y=%.4f->%.4f z=%.4f->%.4f PT=%i'\
                        %(x_min, x_max, y_min, y_max, z_min, z_max, part_type))
    
            f = h5py.File(self.fname, "r")

            # Do we have top level cell information?
            if 'Cells' in f:
                mask = self.find_select_region_cells(f, x_min, x_max, y_min, y_max, z_min, z_max)
                just_load_all = False
            else:
                just_load_all = True
            f.close()
        
        if self.comm_size > 1:
            mask = self.comm.bcast(mask)
            just_load_all = self.comm.bcast(just_load_all)   
 
        # Output dict.
        self.region_data = {}

        # No top level cell information, just load everything.
        if just_load_all:
            if self.verbose: self.message("Just loading all cells.")

            self.region_data['lefts'] = []
            self.region_data['rights'] = []
            self.region_data['files'] = []
            self.region_data['num_to_load'] = []

            # Loop over each file part to index them.
            if self.verbose: self.message("Indexing files.")
            lo_ranks = np.arange(0, self.comm_size + self.max_concur_io, self.max_concur_io)[:-1]
            hi_ranks = np.arange(0, self.comm_size + self.max_concur_io, self.max_concur_io)[1:]
            for lo, hi in zip(lo_ranks, hi_ranks):
                for this_file_i in range(self.HEADER['NumFilesPerSnapshot']):
                    if not (self.comm_rank >= lo and self.comm_rank < hi): continue

                    if self.comm_size > 1:
                        if this_file_i % self.comm_size != self.comm_rank: continue
                    tmp_f = h5py.File(self.get_filename(this_file_i), 'r')
                    tmp_num_this_file = tmp_f['Header'].attrs["NumPart_ThisFile"][part_type]
                    if tmp_num_this_file == 0: continue
                    self.region_data['num_to_load'].append(tmp_num_this_file)
                    self.region_data['lefts'].append(0)
                    self.region_data['rights'].append(tmp_num_this_file)
                    self.region_data['files'].append(this_file_i)
                    tmp_f.close()

                if self.comm_size > 1: self.comm.barrier()

            self.region_data['total_num_to_load'] = self.HEADER['NumPart_Total'][part_type]
            for att in ['lefts', 'rights', 'num_to_load', 'files']:
                if self.comm_size > 1:
                    self.region_data[att] = \
                        np.concatenate(self.comm.allgather(self.region_data[att]))
                self.region_data[att] = np.array(self.region_data[att], dtype='i8')        
            
            # Make sure we found all the particles we intended to.
            assert np.sum(self.region_data['num_to_load']) \
                    == self.region_data['total_num_to_load'], 'Error loading region, count err'

            # We have top level cell information, just load selected region.
        elif len(mask[0]) > 0:
            if self.comm_rank == 0:
                f = h5py.File(self.fname, "r")

                if "Cells/OffsetsInFile/" in f:
                    offsets = f["Cells/OffsetsInFile/PartType%i"%part_type][mask]
                else:
                    offsets = f["Cells/Offset/PartType%i"%part_type][mask]
                counts = f["Cells/Counts/PartType%i"%part_type][mask]
                files = f["Cells/Files/PartType%i"%part_type][mask]
                f.close()
   
                # Only interested in cells with at least 1 particle.
                mask = np.where(counts > 0)
                offsets = offsets[mask]
                counts = counts[mask]
                files = files[mask]    

                # Sort by file number then by offsets.
                mask = np.lexsort((offsets, files))
                offsets = offsets[mask]
                counts = counts[mask]
                files = files[mask]

                if self.verbose:
                    print('%i cells selected from %i file(s).'%(len(offsets),len(np.unique(files))))
                
                # Case of no cells.
                if len(offsets) == 0:
                    raise Exception('No particles found in selected region.')
                # Case of one cell.
                elif len(offsets) == 1:
                    self.region_data['files'] = [files[0]]
                    self.region_data['lefts'] = [offsets[0]]
                    self.region_data['rights'] = [offsets[0] + counts[0]]
                # Case of multiple cells.
                else:
                    self.region_data['lefts'] = []
                    self.region_data['rights'] = [] 
                    self.region_data['files'] = []

                    buff = 0
                    for i in range(len(offsets)-1):
                        if offsets[i] + counts[i] == offsets[i+1]:
                            buff += counts[i]
    
                            if i == len(offsets)-2:
                                self.region_data['lefts'].append(offsets[i+1] - buff)
                                self.region_data['rights'].append(offsets[i+1] + counts[i+1])
                                self.region_data['files'].append(files[i+1])
                        else:
                            self.region_data['lefts'].append(offsets[i] - buff)
                            self.region_data['rights'].append(offsets[i]+counts[i])
                            self.region_data['files'].append(files[i])
                            buff = 0
    
                            if i == len(offsets)-2:
                                self.region_data['lefts'].append(offsets[i+1] - buff)
                                self.region_data['rights'].append(offsets[i+1] + counts[i+1])
                                self.region_data['files'].append(files[i+1])
    
                for tmp_att in self.region_data.keys():
                    self.region_data[tmp_att] = np.array(self.region_data[tmp_att])
                self.region_data['total_num_to_load'] = np.sum(counts)
                self.region_data['num_to_load'] =\
                        self.region_data['rights'] - self.region_data['lefts']

                # Make sure we found all the particles we intended to.
                assert np.sum(self.region_data['num_to_load']) \
                        == self.region_data['total_num_to_load'], 'Error loading region, count err'

            if self.comm_size > 1:
                self.region_data = self.comm.bcast(self.region_data)
        
        else:
            raise Exception('No particles found in selected region.')
        
        self.region_selected_on = part_type

    def get_filename(self, fileno):
        """ For a given partnumber, whats the filename. """
        if self.HEADER['NumFilesPerSnapshot'] > 1:
            return self.fname.split('.')[:-2][0] + '.%i.hdf5'%fileno
        else:
            return self.fname

    def read_dataset(self, parttype, att, physical_cgs=False):
        assert self.HEADER['NumPart_Total'][parttype] > 0,\
            'No particles of PT=%i found in %s. [%s]'%(parttype,self.fname,
            self.HEADER['NumPart_Total'])
        assert self.split_selection_called, 'Need to call split selection first'
        assert self.region_selected_on == parttype,\
                'Selected region on PT=%i but trying to read PT=%i'%(
                        self.region_selected_on, parttype)

        if self.HEADER['NumFilesPerSnapshot'] > 1:
            if self.distributed_format == 'non_collective':
                return self.read_dataset_distributed(parttype, att, physical_cgs)
            else:
                return self.read_dataset_collective(parttype, att, physical_cgs)
        else:
            return self.read_dataset_collective(parttype, att, physical_cgs)

    def read_dataset_distributed(self, parttype, att, physical_cgs):
        """ Each core reads whole file parts. """

        # Get dtype and shape of dataset.
        if self.comm_rank == 0:
            f = h5py.File(self.fname, 'r')
            shape = f['PartType%i/%s'%(parttype, att)].shape
            dtype = f['PartType%i/%s'%(parttype, att)].dtype
            f.close()
        else:
            shape = None
            dtype = None

        if self.comm_size > 1:
            shape = self.comm.bcast(shape)
            dtype = self.comm.bcast(dtype)

        # Number of particles this core is loading.
        tot = np.sum(self.index_data['num_to_load'])
        num_files = len(self.index_data['num_to_load'])
        
        if self.verbose:
            print('[Rank %i] Loading %i particles from %i files.'%(self.comm_rank, tot, num_files))

        # Set up return array.
        if len(shape) > 1:
            return_array = np.empty((tot,shape[1]), dtype=dtype)
            byte_size = dtype.itemsize * shape[1]
        else:
            byte_size = dtype.itemsize
            return_array = np.empty(tot, dtype=dtype)
        return_array = return_array.astype(return_array.dtype.newbyteorder("="))

        # Loop over each file part and load.
        count = 0
        lo_ranks = np.arange(0, self.comm_size + self.max_concur_io, self.max_concur_io)[:-1]
        hi_ranks = np.arange(0, self.comm_size + self.max_concur_io, self.max_concur_io)[1:]
        for lo, hi in zip(lo_ranks, hi_ranks):

            for j, fileno in enumerate(self.index_data['files']):
  
                if not (self.comm_rank >= lo and self.comm_rank < hi): continue
 
                if self.verbose:
                    print('[Rank %i] Loading file %i.'%(self.comm_rank, fileno))
 
                f = h5py.File(self.get_filename(fileno), 'r')
    
                # Loop over each left and right block for this file.
                for l, r in zip(self.index_data['lefts'][j], self.index_data['rights'][j]):
                    this_count = r-l
                    
                    # Can't read more than <max_size_to_read_at_once> at once. 
                    # Need to chunk it.
                    num_chunks = \
                        int(np.ceil((this_count * byte_size) / self.max_size_to_read_at_once))
                    num_per_cycle = np.tile(this_count // num_chunks, num_chunks)
                    mini_lefts = np.cumsum(num_per_cycle) - num_per_cycle[0]
                    mini_rights = np.cumsum(num_per_cycle)
                    num_per_cycle[-1] += this_count % num_chunks
                    mini_rights[-1] += this_count % num_chunks
                    assert np.sum(mini_rights - mini_lefts) == this_count,\
                        'Minis dont add up %i != this_count=%i'%\
                            (np.sum(mini_rights - mini_lefts), this_count)
    
                    # Loop over each 2 Gb chunk.
                    for i in range(num_chunks):
    
                        this_l_return = count + mini_lefts[i]
                        this_r_return = count + mini_rights[i]
                        
                        this_l_read = l + mini_lefts[i]
                        this_r_read = l + mini_rights[i]
    
                        return_array[this_l_return:this_r_return] \
                                = f['PartType%i/%s'%(parttype, att)][this_l_read:this_r_read]
                    
                    count += this_count
    
                f.close()
            if self.comm_size > 1: self.comm.barrier()

        #if physical_cgs: return_array *= physical_fac

        # Apply coordinates offset.
        if att == 'Coordinates':
            if 'CoordinatesOffset' in self.HEADER.keys():
                return_array += self.HEADER['CoordinatesOffset']

        return return_array

    def read_dataset_collective(self, parttype, att, physical_cgs):
        file_offset_count = 0

        # Loop over each file part.
        for j, fileno in enumerate(self.index_data['files']):

            # When there is so few particles (less than 100 per core), only core 0 will read.
            if self.comm_rank > 0 and self.region_data['total_num_to_load'] <= 100*self.comm_size:
                continue

            # How many particles are we loading (on this core) from this file part?
            if self.comm_size > 1 and self.region_data['total_num_to_load'] > 100*self.comm_size:
                tot = self.comm.allreduce(np.sum(self.index_data['num_to_load'][j]))
                if self.verbose:
                    self.message('[%s] Collective loading %i particles from file %i.'\
                        %(att, tot, fileno))
            else:
                tot = np.sum(self.index_data['num_to_load'][j])
                if self.verbose:
                    print('[%s] Serial loading %i particles from file %i.'%(att, tot, fileno))
            
            # Open the hdf5 file.
            if self.comm_size > 1 and self.region_data['total_num_to_load'] > 100*self.comm_size:
                f = h5py.File(self.get_filename(fileno), 'r', driver='mpio', comm=self.comm)
                f.atomic = True
            else:
                f = h5py.File(self.get_filename(fileno), 'r')

            # First time round we setup the return array.
            if j == 0:
                shape = f['PartType%i/%s'%(parttype, att)].shape
                dtype = f['PartType%i/%s'%(parttype, att)].dtype

                if len(shape) > 1:
                    return_array = np.empty((np.sum(self.index_data['total_num_to_load']),shape[1]),
                            dtype=dtype)
                    byte_size = dtype.itemsize * shape[1]
                else:
                    byte_size = dtype.itemsize
                    return_array = np.empty(np.sum(self.index_data['total_num_to_load']),
                            dtype=dtype)
                return_array = return_array.astype(return_array.dtype.newbyteorder("="))

                # What do we multiply by to get physical cgs?
                if physical_cgs:
                    attr_name = \
                        "Conversion factor to physical CGS (including cosmological corrections)"
                    physical_fac = f['PartType%i/%s'%(parttype, att)].attrs.get(attr_name)

            # Populate return array.
            count = 0

            # Loop over each left and right block for this file.
            for l, r in zip(self.index_data['lefts'][j], self.index_data['rights'][j]):
                this_count = r-l
                
                # Can't read more than <max_size_to_read_at_once> at once. 
                # Need to chunk it.
                num_chunks = int(np.ceil((this_count * byte_size) / self.max_size_to_read_at_once))
                num_per_cycle = np.tile(this_count // num_chunks, num_chunks)
                mini_lefts = np.cumsum(num_per_cycle) - num_per_cycle[0]
                mini_rights = np.cumsum(num_per_cycle)
                num_per_cycle[-1] += this_count % num_chunks
                mini_rights[-1] += this_count % num_chunks
                assert np.sum(mini_rights - mini_lefts) == this_count,\
                    'Minis dont add up %i != this_count=%i'%\
                        (np.sum(mini_rights - mini_lefts), this_count)
   
                # Loop over each 2 Gb chunk.
                for i in range(num_chunks):

                    this_l_return = file_offset_count + count + mini_lefts[i]
                    this_r_return = file_offset_count + count + mini_rights[i]
                    
                    this_l_read = l + mini_lefts[i]
                    this_r_read = l + mini_rights[i]

                    return_array[this_l_return:this_r_return] \
                            = f['PartType%i/%s'%(parttype, att)][this_l_read:this_r_read]
                
                count += this_count

            f.close()

            # Keep track of offset by reading multiple files in return array.
            file_offset_count += count

        # Get return array dtype for low particle case.
        if self.comm_rank == 0:
            return_array_dtype = return_array.dtype
        else:
            return_array_dtype = None
        if self.comm_size > 1: return_array_dtype = self.comm.bcast(return_array_dtype)

        # When there is so few particles (less than 100 per core), only core 0 will read.
        if self.comm_rank > 0 and self.region_data['total_num_to_load'] <= 100*self.comm_size:
            return np.array([], dtype=return_array_dtype)
        else:
            # Convert to physical cgs?
            if physical_cgs: return_array *= physical_fac

            # Apply coordinates offset.
            if att == 'Coordinates':
                if 'CoordinatesOffset' in self.HEADER.keys():
                    return_array += self.HEADER['CoordinatesOffset']

            return return_array

    def read_header(self):
        """ Get information from the header and cosmology. """
        f = h5py.File(self.fname, "r")

        # Load header information.
        self.HEADER = {}
        for att in f['Header'].attrs.keys():
            self.HEADER[att] = f['Header'].attrs.get(att)

            # Convert single value arrays to scalars.
            if type(self.HEADER[att]) == np.ndarray:
                if len(self.HEADER[att]) == 1:
                    self.HEADER[att] = self.HEADER[att][0]

        # Convert total number of particles to unsigned I64.
        if 'NumPart_Total' in self.HEADER.keys():
            self.HEADER['NumPart_Total'] = np.array(self.HEADER['NumPart_Total'], dtype=np.uint64)

        # Deal with highword.
        if 'NumPart_Total_HighWord' in self.HEADER.keys():
            for i in range(len(self.HEADER['NumPart_Total'])):
                self.HEADER['NumPart_Total'][i] = self.HEADER['NumPart_Total'][i] + \
                        (self.HEADER['NumPart_Total_HighWord'][i] * 2**32)
                self.HEADER['NumPart_Total_HighWord'][i] = 0

        # Assume boxsize is equal in all dimensions.
        if 'BoxSize' in self.HEADER.keys():
            if type(self.HEADER['BoxSize']) == list or type(self.HEADER['BoxSize']) == np.ndarray:
                if len(self.HEADER['BoxSize']) == 3:
                    self.HEADER['BoxSize'] = float(self.HEADER['BoxSize'][0])

        # Load cosmology information.
        if 'Cosmology' in f:
            self.COSMOLOGY = {}
            for att in f['Cosmology'].attrs.keys():
                self.COSMOLOGY[att] = f['Cosmology'].attrs.get(att)

        # Load parameters.
        if 'Parameters' in f:
            self.PARAMETERS = {}
            for att in f['Parameters'].attrs.keys():
                self.PARAMETERS[att] = f['Parameters'].attrs.get(att)
        f.close()

if __name__ == '__main__':
    from mpi4py import MPI
    import time

    comm = MPI.COMM_WORLD
    comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)

    tic = time.time()
    snap = '/cosma6/data/dp004/rttw52/SibeliusOutput/Sibelius_200Mpc_1/joined_hsmls_index/volume1/0199.hdf5'
    snap = '/cosma6/data/dp004/rttw52/SibeliusOutput/Sibelius_200Mpc_256/snapshots/Sibelius_200Mpc_256_0199/Sibelius_200Mpc_256_0199.0.hdf5'
    x = read_swift(snap, comm=comm, verbose=True)
    x.select_region(1,300,700,300,700,300,700)
    x.split_selection()
    ids = x.read_dataset(1, 'ParticleIDs')
    coords = x.read_dataset(1, 'Coordinates')
    mask = np.where(ids == 8382864541576428481)
    comm.barrier()
    print(coords[mask], ids[mask])
