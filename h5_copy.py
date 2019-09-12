def _copy_group_(src, dst, name=None):
    '''
    Copies one h5py Group and all of its members into an indicated destination
    onject (either a h5py File or Group object)
    
    Input:
    --------
        src : h5py.Group
            Group to be copied to new location
        dst : h5py.Group or h5py.File
            Destination for copied group. src will be copied as a new group
            created within dst, along with all subgroups and datasets of src.
    '''
    keys = [key for key in src.keys()]
    if name == None:
        name = src.name.split('/')[-1]
    new_grp = dst.create_group(name)
    for key in keys:
        item = src[key]
        if type(item) == h5py._hl.dataset.Dataset:
            CoReg._copy_dataset_(item, new_grp)
        elif type(item) == h5py._hl.group.Group:
            CoReg._copy_group_(item, new_grp)
    for key, value in src.attrs.items():
        new_grp.attrs[key] = value
    return
	
def _copy_dataset_(src, dst, name=None):
    '''
    Copies one h5py Dataset to a new location, either within the same file
	or in a separate file. 
    
    Input:
    --------
        src : h5py.Dataset
            Dataset to be copied to new location
        dst : h5py.Group or h5py.File
            Destination for copied dataset. src will be copied as a new dataset
            created within dst.
    '''
    import sys
    import psutil
    
    def copy_blockwise(n_blocks):
        
        block_height = int(np.floor(src.shape[0]/n_blocks))
        block_edges = np.linspace(0, n_blocks-1, n_blocks, dtype=int)
        for lower in block_edges:
            lower = lower * block_height
            upper = lower + block_height
            dst_dset[lower:upper] = src[lower:upper]
        dst_dset[upper:] = src[upper:]
        return
        
    src_dtype = src.dtype
    src_shape = src.shape
    src_chunk = src.chunks
    src_dim = len(src_shape)
    src_sample = src[0]
    while True:
        try:
            if len(src_sample) > 1:
                src_sample = src_sample[0]
            else:
                src_unitsize = sys.getsizeof(src_sample)
                break
        except TypeError:
            src_unitsize = sys.getsizeof(src_sample)
            break
    dset_size = src_unitsize
    for dim in src_shape:
        dset_size = dset_size * dim
    available_memory = psutil.virtual_memory().available
    usable_memory = int(available_memory*0.75)
    memory_ratio = dset_size / usable_memory
        
    if name == None:
        name = src.name.split('/')[-1]
    
    dst_dset = dst.create_dataset(name, shape=src.shape, chunks=src.chunks, dtype=src.dtype)
    if memory_ratio > 1:
        n_blocks = 0
        while memory_ratio > 1:
            n_blocks += 1
            subset_size = dset_size / n_blocks
            memory_ratio = subset_size / usable_memory
        copy_blockwise(n_blocks)
    else:
        n_blocks = 20
        copy_blockwise(n_blocks)
       
    for key, value in src.attrs.items():
        new_grp.attrs[key] = value
            
    return