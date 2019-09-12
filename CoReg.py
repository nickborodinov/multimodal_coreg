# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:12:11 2019

@author: s4k
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import time
import pyUSID as usid

def corr_indexing(im):
    x, y = im.shape
    corrected = np.zeros_like(im)
    i = 0
    j = 0
    for pixel in im.flatten():
        corrected[i,j] = pixel
        i += 1
        if i == x:
            i=0
            j+=1
    return corrected

def find_nearest_member(container, query):
    '''
    Finds the member of a container whose value is nearest to query. Returns
    index of nearest value within container. Intended to be used when 
    list.index(query) is, for whatever reason, not a viable option for locating
    the desired value within the container.
    
    Input:
    --------
    container : container variable (eg list, tuple, set, Numpy array)
        The container to be searched by the function
    query : number (eg int or float)
        Value to be searched for within container
        
    Output:
    --------
    mindex : int
        Index of item in container whose value most nearly matches query
    '''
    try:
        diffs = abs(container - query)
    except:
        diffs = []
        for entry in container:
            difference = entry - query
            diffs.append(abs(difference))
    minimum = min(diffs)
    mindex = list(diffs).index(minimum)
    return mindex

def q_test(data, value, confidence=95):
    '''
    Performs Dixon's Q Test for outliers on a value within a dataset.
    
    Input:
    --------
        data : array-like
            Dataset within which value is suspected of being an outlier
        value : int or float
            Value suspected of being an outlier within data
        confidence : int (90, 95, or 99)
            Confidence interval, as a percentage, to consider for outlier testing
    '''
    q90 = [0.941, 0.765, 0.642, 0.56, 0.507, 0.468, 0.437,
       0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
       0.32, 0.313, 0.306, 0.3, 0.295, 0.29, 0.285, 0.281,
       0.277, 0.273, 0.269, 0.266, 0.263, 0.26]
    q95 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
       0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
       0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
       0.308, 0.305, 0.301, 0.29]
    q99 = [0.994, 0.926, 0.821, 0.74, 0.68, 0.634, 0.598, 0.568,
       0.542, 0.522, 0.503, 0.488, 0.475, 0.463, 0.452, 0.442,
       0.433, 0.425, 0.418, 0.411, 0.404, 0.399, 0.393, 0.388,
       0.384, 0.38, 0.376, 0.372]
    Q90 = {n:q for n,q in zip(range(3,len(q90)+1), q90)}
    Q95 = {n:q for n,q in zip(range(3,len(q95)+1), q95)}
    Q99 = {n:q for n,q in zip(range(3,len(q99)+1), q99)}
    #Determine range of input dataset
    x_range = max(data) - min(data)
    #Find distance of value to nearest data point in data
    neardex = find_nearest_member(value, data)
    nearest = data[neardex]
    gap = np.abs(value-nearest)
    #Calculate critical q
    q = gap / x_range
    if confidence == 95: q_table = Q95
    elif confidence == 90: qtable = Q90
    elif confidence == 99: qtable = Q99
    else: raise KeyError('Invalid value selected for confidence')
    #Compare to H in Q table
    h = q_table[len(data)]
    if q > h: result = True
    else: result = False
    return result

class CoReg:
    '''
    Handle for manipulation and analysis of co-registered multimodal
    imaging datasets.
    '''
    
    def __init__(self, merged_fname):
        self.merged_fname = merged_fname        #Absolute filename of the merged HDF5 dataset
        if not os.path.isfile(self.merged_fname):
            hf = self.open_merged('w')
            hf.close()
        return
    
    def add_dataset(self, main_data, main_data_name, quantity, units, pos_dims,\
                    spec_dims, anchors, main_dset_attrs=None, h5_pos_inds=None, h5_pos_vals=None,\
                    h5_spec_inds=None, h5_spec_vals=None, aux_spec_prefix='Spectroscopic_',\
                    aux_pos_prefix='Position_', verbose=False, **kwds):
        hf = self.open_merged('r+')
        new_grp = hf.create_group(main_data_name)
        usid.hdf_utils.write_main_dataset(
                h5_parent_group=new_grp,
                main_data=main_data,
                main_data_name='Main',
                quantity=quantity,
                units=units,
                pos_dims=pos_dims,
                spec_dims=spec_dims, 
                main_dset_attrs=main_dset_attrs, 
                h5_pos_inds=h5_pos_inds, 
                h5_pos_vals=h5_pos_vals,
                h5_spec_inds=h5_spec_inds,
                h5_spec_vals=h5_spec_vals)
        hf.flush()
        hf[main_data_name].create_dataset('anchors', data=np.array(anchors, dtype=np.float32))
        hf.close()
        return
    
    def batch_add_dataset(self, main_data, main_data_name, quantity, units, pos_dims,\
                          spec_dims, anchors, main_dset_attrs=None, h5_pos_inds=None,\
                          h5_pos_vals=None,h5_spec_inds=None, h5_spec_vals=None,\
                          aux_spec_prefix='Spectroscopic_',aux_pos_prefix='Position_',\
                          verbose=False, **kwds):
        hf = self.open_merged('r+')
        try: new_grp = hf.create_group(main_data_name)
        except ValueError: new_grp = hf[main_data_name]
        new_dset = usid.hdf_utils.write_main_dataset(
                h5_parent_group=new_grp,
                main_data=main_data.shape,
                main_data_name='Main',
                quantity=quantity,
                units=units,
                pos_dims=pos_dims,
                spec_dims=spec_dims, 
                main_dset_attrs=main_dset_attrs, 
                h5_pos_inds=h5_pos_inds, 
                h5_pos_vals=h5_pos_vals,
                h5_spec_inds=h5_spec_inds,
                h5_spec_vals=h5_spec_vals,
                dtype=main_data.dtype)
        if 'batchsize' in kwds: batchsize = kwds['batchsize']
        else: batchsize = 10000
        n_batches = (main_data.shape[0]//batchsize) + 1
        for i in range(main_data.shape[0]//batchsize):
            low = i * batchsize
            high = low + batchsize
            new_dset[low:high] = main_data[low:high]
            print('Finished batch %i of %i'%(i+1, n_batches))
        new_dset[high:] = main_data[high:]
        print('Finished batch %i of %i'%(i+2, n_batches))
        hf.flush()
        hf[main_data_name].create_dataset('anchors', data=np.array(anchors, dtype=np.float32))
        hf.close()
        return
    
    def set_anchors(self, grp, anchors):
        grp['anchors'][:] = np.array(anchors, dtype=float32)
        return
    
    def open_merged(self, rw='r'):
        return h5py.File(self.merged_fname, rw)
    
    def get_image(self, grp, spec, tol=0.001, *args, **kwds):
        data = usid.USIDataset(grp['Main'])
        s_labels = data.spec_dim_labels
        if 's_dim' not in kwds: 
            label = s_labels[0]
        else: 
            s_dim = kwds['s_dim']
            if type(s_dim) is str: label = s_dim
            else: label = s_labels[int(s_dim)]
        bar = data.get_spec_values(label)
            
        try:
            bounds_idx = (find_nearest_member(bar, spec-(spec*tol)), find_nearest_member(bar, spec+(spec*tol)))
            if bounds_idx[1]==bounds_idx[0]: bounds_idx[1] = bounds_idx[0]+1
        except:
            grpname = grp.name.split('/')[-1]
            print('No image could be obtained for: %s at %f'%(grpname, spec))
            return None
        hit_stack, result = data.slice({label:slice(bounds_idx[0],bounds_idx[1])})
        hit_mz = bar[bounds_idx[0]:bounds_idx[1]]
        mz_stack = np.empty_like(hit_stack)
        
        for i, v in enumerate(hit_mz):
            mz_stack[...,i] = v
        if 'norm' in kwds:
            #apply normalization
            norm_mode = kwds['norm']
            norm_mode_options = [key.upper() for key in grp['norm'].keys()]
            if norm_mode is None: norm_factors = np.ones(data.shape[:-1])
            elif norm_mode.upper() in norm_mode_options:
                norm_factors = grp['norm'][norm_mode.upper()][:]
            else: raise KeyError('Selected normalization mode has not been calculated.')
        else: norm_factors = np.ones(data.shape[:-1])
        
        try: image = np.trapz(hit_stack, mz_stack, axis=2)
        except IndexError: pass
        image = np.multiply(image, 1/norm_factors.reshape(image.shape))
        return image
    
    def get_coregistered_image(self, src, dst, spec, tol=0.001, *args, **kwds):
        src_image = self.get_image(src, spec, tol, args, kwds)
        src_anchors = src['anchors'][:]
        dst_anchors = dst['anchors'][:]
        affine = cv2.getAffineTransform(src_anchors, dst_anchors)
        dst_image = cv2.warpAffine(src_image, affine, src_image.shapehf)
        return dst_image
    
    def get_spectrum(self, grp, coords, norm=None):
        #!!! No normalization functionality
        #!!! Currently broken, not made compatible with USID
        data = grp['intensity']
        bar = grp['axis'][:]
        norm_grp = grp['norm']
        cx, cy, ct = data.chunks
        shape = data.shape
        valid_coords = []
        for x, y in coords:
            if x < shape[0] and y < shape[1]:
                valid_coords.append((x, y))
        spectra = np.array([np.empty_like(bar) for x, y in valid_coords])
        for i in range(shape[2]//ct):
            low = ct * i
            high = low + ct
            stack = data[..., low:high]
            for idx, (x, y) in enumerate(valid_coords):
                spectra[idx, low:high] = stack[x, y]
        stack = data[:, high:]
        for idx, (x, y) in enumerate(valid_coords):
            spectra[idx, high:] = stack[x, y]
        return spectra
    
    def scale_image(self, src, dst):
        return image
    
    @staticmethod
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
    
    @staticmethod
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
            
        return dst_dset
    
    def _estimate_mask_(self):
        
        hf = self.open_merged(self.merged_fname)
        maldi_dataset = hf['MALDI']['intensity']
        sims_dataset = hf['SIMS']['intensity']
        x, y, z = maldi_dataset.shape
        dummy = np.ones(shape=(x, y), dtype=np.float32)
        x, y, z = sims_dataset.shape
        dummy = self._scale_maldi_image_(dummy, (y, x))
        plt.imshow(dummy)
        return
